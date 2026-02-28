import torch
import argparse
from torchvision.utils import save_image
from models import DiT_models #

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

    # Load Custom DiT in Pixel Space (in_channels=3)
    print(f"Loading DiT model from {args.ckpt}...")
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=3,
        num_classes=args.num_classes
    ).to(device)
    
    # Load checkpoint
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
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

    # 3. Initialize pure noise (t=0) in PIXEL SPACE
    # Shape is (N, 3, 64, 64)
    z = torch.randn(args.n, 3, args.image_size, args.image_size, device=device)
    dt = 1.0 / args.num_steps

    print(f"Generating {args.n} images with attributes: {args.attributes}")
    print(f"Using Vanilla CFG Scale: {args.cfg_scale}, Steps: {args.num_steps}")

    # 4. The Euler ODE Solver (Flow Matching generation)
    for i in range(args.num_steps):
        t_val = i / args.num_steps
        t = torch.full((args.n,), t_val, device=device)

        # Double batch for CFG efficiency
        z_batched = torch.cat([z, z], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        y_batched = torch.cat([y_cond, y_uncond], dim=0)

        # force_drop_ids triggers null_token for the second half of the batch
        drop_ids = torch.cat([
            torch.zeros(args.n, dtype=torch.bool, device=device), 
            torch.ones(args.n, dtype=torch.bool, device=device)
        ])

        # Forward pass
        v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
        
        # DiT outputs 2*in_channels (6 for RGB) because learn_sigma=True
        v_batched, _ = v_batched.chunk(2, dim=1)
        
        # Split into conditional and unconditional velocities
        v_cond, v_uncond = v_batched.chunk(2, dim=0)
        
        # Apply standard Vanilla CFG
        v_cfg = v_uncond + args.cfg_scale * (v_cond - v_uncond)
        
        # Euler step forward: x_{t+dt} = x_t + v_t * dt
        z = z + v_cfg * dt

    # 5. Process final pixels (No VAE decoding needed)
    # Unnormalize from [-1, 1] to [0, 1]
    samples = (z + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # 6. Save the image grid
    grid_size = int(args.n ** 0.5)
    safe_name = "_".join(args.attributes) if args.attributes else "unconditional"
    filename = f"vanilla_cfg_{safe_name}.png"
    
    save_image(samples, filename, nrow=grid_size)
    print(f"Saved generated grid to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2") #
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your trained .pt checkpoint")
    parser.add_argument("--attributes", type=str, nargs='+', help="List of attributes (e.g., Male Eyeglasses Smiling)")
    parser.add_argument("--n", type=int, default=16, help="Number of images to generate in the grid")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of Euler integration steps")
    parser.add_argument("--seed", type=int, default=50, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)