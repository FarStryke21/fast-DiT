import torch
import argparse
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models import DiT_models # Assumes your custom DiT is still in models.py

# Map the exact 40 CelebA attributes
ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 
    'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]
ATTR_MAP = {name: i for i, name in enumerate(ATTR_NAMES)}

def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

    # Load Custom DiT
    print(f"Loading DiT model from {args.ckpt}...")
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    # Load checkpoint
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in state_dict:
        model.load_state_dict(state_dict["ema"])
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # 1. Build the conditional tensor based on user args
    y_cond = torch.zeros(args.n, args.num_classes, device=device)
    if args.attributes:
        for attr in args.attributes:
            if attr in ATTR_MAP:
                y_cond[:, ATTR_MAP[attr]] = 1.0
            else:
                print(f"Warning: '{attr}' is not a valid attribute. Ignoring.")
                
    # 2. Build the unconditional tensor (all zeros)
    y_uncond = torch.zeros(args.n, args.num_classes, device=device)

    # 3. Initialize pure noise (t=0)
    z = torch.randn(args.n, model.in_channels, latent_size, latent_size, device=device)
    dt = 1.0 / args.num_steps

    print(f"Generating {args.n} images with attributes: {args.attributes}")
    print(f"Using Method: {args.mp_method.upper()} | CFG Scale: {args.cfg_scale} | Steps: {args.num_steps}")
    print(f"Projection Window: t={args.t_proj_start} to t={args.t_proj_end}")

    # 4. The Euler ODE Solver (Flow Matching generation)
    for i in range(args.num_steps):
        # Create time tensor for this step (from 0.0 to 1.0)
        t_val = i / args.num_steps
        t = torch.full((args.n,), t_val, device=device)

        # We double the batch to process conditional and unconditional at the same time
        z_batched = torch.cat([z, z], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        y_batched = torch.cat([y_cond, y_uncond], dim=0)

        drop_ids = torch.cat([
            torch.zeros(args.n, dtype=torch.bool, device=device), 
            torch.ones(args.n, dtype=torch.bool, device=device)
        ])

        # Forward pass
        v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
        
        # Slice off the extra variance channels
        v_batched, _ = v_batched.chunk(2, dim=1)
        
        # Split back into conditional and unconditional velocities
        v_cond, v_uncond = v_batched.chunk(2, dim=0)
        
        # Apply standard Vanilla CFG
        v_cfg = v_uncond + args.cfg_scale * (v_cond - v_uncond)

        # ---------------------------------------------------------
        # PREDICTOR STEP
        # ---------------------------------------------------------
        x = z + v_cfg * dt  # Euler step forward
        
        # Setup time and drop_ids for the corrector evaluations
        t_next_val = (i + 1) / args.num_steps
        t_next = torch.full((args.n,), t_next_val, device=device)
        drop_ids_proj = torch.ones(args.n, dtype=torch.bool, device=device)

        # ---------------------------------------------------------
        # CORRECTOR STEP (Time-Gated Manifold Projection)
        # ---------------------------------------------------------
        if args.t_proj_start <= t_val <= args.t_proj_end:
            if args.mp_method == "standard":
                # Standard iterative projection (Requires K-1 extra evaluations)
                for k in range(1, args.proj_K):
                    v_proj = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj) 
                    v_proj, _ = v_proj.chunk(2, dim=1)
                    
                    # Nudge the latent back toward the unconditional manifold
                    x = x + (v_proj - v_uncond) * dt * 0.5

            elif args.mp_method == "anderson":
                # Anderson Accelerated projection (Requires exactly 2 extra evaluations)
                # Eval 1
                v_proj_1 = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                v_proj_1, _ = v_proj_1.chunk(2, dim=1)
                
                g_1 = x + (v_proj_1 - v_uncond) * dt * 0.5
                f_1 = g_1 - x

                # Eval 2 
                v_proj_2 = model(g_1, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                v_proj_2, _ = v_proj_2.chunk(2, dim=1)
                
                g_2 = g_1 + (v_proj_2 - v_uncond) * dt * 0.5
                f_2 = g_2 - g_1

                # Anderson Mixing
                delta_f = f_2 - f_1
                f_2_flat = f_2.view(args.n, -1)
                delta_f_flat = delta_f.view(args.n, -1)
                
                numerator = torch.sum(f_2_flat * delta_f_flat, dim=1)
                denominator = torch.sum(delta_f_flat * delta_f_flat, dim=1) + 1e-8
                alpha = (numerator / denominator).view(args.n, 1, 1, 1)
                
                # Extrapolate
                x = g_2 - alpha * (g_2 - g_1)
        
        # Set the final corrected latent for the next ODE loop
        z = x

    # 5. Decode the final latent back into pixels
    print("Decoding latents...")
    z = z / 0.18215
    samples = vae.decode(z).sample
    
    # 6. Save the image grid
    grid_size = int(args.n ** 0.5)
    safe_name = "_".join(args.attributes) if args.attributes else "unconditional"
    filename = f"cfg_mp_{args.mp_method}_{safe_name}.png"
    
    save_image(samples, filename, nrow=grid_size, normalize=True, value_range=(-1, 1))
    print(f"Saved generated grid to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[64, 256, 512], default=64)
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your trained .pt checkpoint")
    parser.add_argument("--attributes", type=str, nargs='+', help="List of attributes (e.g., Male Eyeglasses Smiling)")
    parser.add_argument("--n", type=int, default=16, help="Number of images to generate in the grid")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of Euler integration steps")
    parser.add_argument("--seed", type=int, default=50, help="Random seed for reproducibility")
    parser.add_argument("--mp-method", type=str, choices=["standard", "anderson"], default="standard", help="Which manifold projection method to use")
    parser.add_argument("--proj-K", type=int, default=3, help="Number of projection steps (only used if mp-method is 'standard')")
    parser.add_argument("--t-proj-start", type=float, default=0.2, help="Start time for the speciation projection window (0.0 to 1.0)")
    parser.add_argument("--t-proj-end", type=float, default=0.7, help="End time for the speciation projection window (0.0 to 1.0)")
    args = parser.parse_args()
    main(args)