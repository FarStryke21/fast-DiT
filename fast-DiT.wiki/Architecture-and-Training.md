# Architecture and Training

## 1. What was changed from upstream DiT

The fork keeps Meta's DiT block structure verbatim (adaLN-Zero, sin-cos frozen positional embeddings, zero-init output layers) and changes four things:

| Change | Where | Why |
|---|---|---|
| **VAE removed — pixel space** | `train.py:169` (commented out), all samplers construct the model with `in_channels=3` | 64×64 is small enough to model directly; removes a decode from the ODE loop and makes the "manifold" the pixel manifold, not a latent one. Commit `4f88c1d` "Remvoe VAE encoding". |
| **Multi-label conditioning** | `models.py:97-124`, wired at `models.py:201` | CelebA has 40 non-exclusive binary attributes; a `nn.Embedding` lookup table cannot represent them. |
| **Flow matching instead of DDPM** | `train.py:251-272` | Straight-line paths give a velocity field with a clean ODE, which is what the corrector operates on. |
| **HF-streamed, RAM-cached CelebA loader** | `dataset.py` | Replaces the pre-extracted `.npy` feature pipeline. |

## 2. `MultiLabelEmbedder` (`models.py:97-124`)

```python
class MultiLabelEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, dropout_prob):
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),   # 40 -> 768
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),   # 768 -> 768
        )
        self.null_token = nn.Parameter(torch.randn(1, in_channels))   # learned, shape (1, 40)
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            if force_drop_ids is not None:
                drop_ids = force_drop_ids
            labels = torch.where(drop_ids[:, None], self.null_token, labels)
        return self.mlp(labels)
```

Points that matter downstream:

- The **null token lives in attribute space (40-dim), not embedding space**, and is passed through the same MLP. It is initialised with `nn.init.normal_(std=0.02)` in `DiT.initialize_weights` (`models.py:231`), overriding the `torch.randn` in the constructor.
- **Dropout is only active when `model.training` is True.** `train.py:229` calls `model.train()` with the comment *"important! This enables embedding dropout for classifier-free guidance"*, and `class_dropout_prob` defaults to `0.1` (`models.py:187`) — never overridden anywhere, so **10% CFG dropout** is what was trained.
- At inference the only way to get the unconditional branch is `force_drop_ids=<ones>`. Passing `y=zeros` with no `force_drop_ids` gives you the *zero attribute vector*, which is a different conditioning signal than the null token — see [Gotchas](Gotchas-and-Known-Issues.md#1-the-uncond-baseline-does-not-use-the-null-token).
- Conditioning enters the network as `c = t_emb + y_emb` (`models.py:279`), then drives every adaLN-Zero modulation. Same as stock DiT.

The original label-lookup `LabelEmbedder` is still present in `models.py:67-94` but is **unused** — nothing constructs it.

## 3. Backbone configuration

`DiT_models["DiT-B/2"]` → `DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12)`, instantiated everywhere as:

```python
DiT_models[args.model](input_size=64, in_channels=3, num_classes=40)
```

- **Tokens**: `(64/2)² = 1024` patches of 2×2×3.
- **`num_classes=40` is passed but ignored** — `DiT.__init__` hard-codes `MultiLabelEmbedder(in_channels=40, ...)` at `models.py:201` and never reads `num_classes` for the embedder. Changing `--num-classes` on the CLI changes nothing about the model; it only changes the width of the `y` tensors the samplers build, which must stay at 40 or the MLP will fail.
- **`learn_sigma=True`** (default, never disabled) ⇒ `out_channels = 6`. Every single call site does `v, _ = v.chunk(2, dim=1)` and discards the second three channels. This is dead capacity in a flow-matching model (there is no variance to learn), costing parameters in `final_layer.linear` and a modest amount of compute.
- **Gradient checkpointing is unconditional** (`models.py:281`): `torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)` runs for every block on every forward, *including inference under `no_grad`*. This slows sampling for no benefit. See [Gotchas](Gotchas-and-Known-Issues.md#4-gradient-checkpointing-is-always-on-including-at-inference).
- `forward_with_cfg` (`models.py:286-302`) is inherited, hard-codes `[:, :3]`, and is **unused** — all samplers do their own batched CFG.

## 4. Training loop (`train.py`)

```
Accelerator(mixed_precision="bf16")
model  = DiT-B/2, input_size=64, in_channels=3
ema    = deepcopy(model), requires_grad=False, decay 0.9999, updated every step
opt    = AdamW(lr=1e-4, weight_decay=0)      # DiT paper defaults
loader = create_dataloader(..., from_hub=True, repo="electronickale/cmu-10799-celeba64-subset")
global_batch_size = 256   (split across accelerator.num_processes)
epochs = 3000 (default; real runs stopped at fixed step counts)
log_every = 100, ckpt_every = 6000
```

Per step:

```python
x_1 = x                                  # image batch, [-1,1]
x_0 = torch.randn_like(x_1)
y   = y.float()                          # (B, 40)
t   = torch.rand(B, 1, 1, 1)
x_t = (1 - t) * x_0 + t * x_1
v_target = x_1 - x_0
v_pred, _ = model(x_t, t.view(-1), y).chunk(2, dim=1)
loss = F.mse_loss(v_pred, v_target)
```

`t` is sampled uniform on `[0,1)` with no logit-normal or cosine reweighting — the plainest possible flow-matching objective.

**Checkpointing.** `{model, ema, opt, args}` saved to `results/{NNN}-DiT-B-2/checkpoints/{step:07d}.pt`, plus `final-checkpoint.pt` at the end. `--resume-from` re-derives `train_steps` from the filename and `start_epoch = train_steps // len(loader)`, then appends to the existing `log.txt`. **All samplers load `state_dict["ema"]` in preference to `state_dict["model"]`** — every reported number is from EMA weights.

**Vestigial code in `train.py`**: `create_diffusion()` is called (`train.py:168`) and never used; `assert args.image_size % 8 == 0` and `latent_size = args.image_size // 8` (`train.py:157-158`) are VAE leftovers with `latent_size` unused; `CustomDataset`, `center_crop_arr`, and the `AutoencoderKL` import are all inert.

## 5. Data pipeline (`dataset.py`)

`CelebADataset` supports HF-hub, HF-`save_to_disk` Arrow, and a local `images/ + attributes.csv` layout. The path actually used by `train.py` is **`from_hub=True` against `electronickale/cmu-10799-celeba64-subset`**.

The distinctive part is that `__init__` **eagerly decodes and preprocesses the entire split into one contiguous RAM tensor** (`dataset.py:68-101`):

```python
static_transform = Resize(64) → CenterCrop(64) → ToTensor() → Normalize(0.5, 0.5)   # → [-1, 1]
self.cached_images = torch.stack([...])   # (N, 3, 64, 64) float32
self.cached_labels = torch.stack([...])   # (N, 40) float32
```

and `__getitem__` becomes a tensor index plus an in-place `torch.flip(image, dims=[2])` with p=0.5 for horizontal-flip augmentation. This trades startup time and RAM (printed as a GB figure at load) for near-zero per-step dataloading cost — the "fast" in fast-DiT, re-derived for this dataset. `_build_transforms()` is left behind as the un-used PIL-transform version.

**Attribute handling.** Attribute columns are everything except `image`/`image_id`, **sorted alphabetically** (`dataset.py:185-187`). Values in `{-1,+1}` are remapped to `{0,1}` via `(attr + 1) // 2`. The sorted-column order is what defines the meaning of each of the 40 slots — and it must match the `ATTR_NAMES` list hard-coded in the samplers and the label order of the ResNet classifier. That list is the canonical CelebA order, which *is* alphabetical:

```
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips,
Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin,
Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open,
Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline,
Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat,
Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young
```

(`5_o_Clock_Shadow` sorts first because `'5' < 'A'`.) This ordering is a silent cross-component contract — if it ever breaks, attribute accuracy collapses while FID stays fine, which is the signature to watch for.

`download_dataset.py` walks the same loader with `shuffle=False, augment=False` and dumps `./data/local_celeba/real_images/{i:06d}.png` plus `attributes.pt` — the **FID reference set** and the **conditioning pool** respectively. Both must come from this one pass or the index alignment between `attributes.pt` and the real images is lost.
