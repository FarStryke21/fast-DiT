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
    print(f"Using Vanilla CFG Scale: {args.cfg_scale}, Steps: {args.num_steps}")

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
        
        # Slice off the extra variance channels (DiT output is 8 channels, Flow Matching uses 4)
        v_batched, _ = v_batched.chunk(2, dim=1)
        
        # Split back into conditional and unconditional velocities
        v_cond, v_uncond = v_batched.chunk(2, dim=0)
        
        # Apply standard Vanilla CFG
        v_cfg = v_uncond + args.cfg_scale * (v_cond - v_uncond)
        
        # Euler step forward
        z = z + v_cfg * dt

    # 5. Decode the final latent back into pixels
    print("Decoding latents...")
    z = z / 0.18215
    samples = vae.decode(z).sample
    
    # 6. Save the image grid
    grid_size = int(args.n ** 0.5)
    safe_name = "_".join(args.attributes) if args.attributes else "unconditional"
    filename = f"vanilla_cfg_{safe_name}.png"
    
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
    args = parser.parse_args()
    main(args)