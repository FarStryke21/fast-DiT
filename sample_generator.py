import torch
import argparse
import os
import json
from tqdm import tqdm
from torchvision.utils import save_image
from models import DiT_models 

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    out_dir = f"samples_{args.method}_w{args.cfg_scale}_steps{args.num_steps}"
    fake_dir = os.path.join(out_dir, "fake")
    os.makedirs(fake_dir, exist_ok=True)

    # Load Model
    print(f"Loading model from {args.ckpt}...")
    model = DiT_models[args.model](input_size=args.image_size, in_channels=3, num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["ema"] if "ema" in state_dict else state_dict)
    model.eval()

    # Load real attributes and sample randomly
    print(f"Loading real attributes from {args.attr_path}...")
    all_real_y = torch.load(args.attr_path)
    
    # Randomly select N attribute combinations
    indices = torch.randperm(len(all_real_y))[:args.num_samples]
    target_conditions = all_real_y[indices]

    total_generated = 0
    nfe_total = 0

    print(f"Starting generation of {args.num_samples} samples using method: {args.method.upper()}")

    with tqdm(total=args.num_samples) as pbar:
        while total_generated < args.num_samples:
            current_batch_size = min(args.batch_size, args.num_samples - total_generated)
            
            # Slice the pre-selected conditions for this batch
            y_cond = target_conditions[total_generated : total_generated + current_batch_size].to(device)
            y_uncond = torch.zeros_like(y_cond).to(device)
            
            if args.method == "uncond":
                y_cond = y_uncond
                
            z = torch.randn(current_batch_size, 3, args.image_size, args.image_size, device=device)
            dt = 1.0 / args.num_steps

            for i in range(args.num_steps):
                t_val = i / args.num_steps
                t = torch.full((current_batch_size,), t_val, device=device)

                # Predictor Step
                if args.method == "uncond":
                    v = model(z, t, y_uncond)
                    v, _ = v.chunk(2, dim=1)
                    x = z + v * dt
                    nfe_total += current_batch_size 
                else:
                    z_batched = torch.cat([z, z], dim=0)
                    t_batched = torch.cat([t, t], dim=0)
                    y_batched = torch.cat([y_cond, y_uncond], dim=0)
                    drop_ids = torch.cat([torch.zeros(current_batch_size, dtype=torch.bool, device=device), 
                                          torch.ones(current_batch_size, dtype=torch.bool, device=device)])

                    v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
                    v_batched, _ = v_batched.chunk(2, dim=1)
                    v_cond, v_uncond_out = v_batched.chunk(2, dim=0)
                    
                    v_cfg = v_uncond_out + args.cfg_scale * (v_cond - v_uncond_out)
                    x = z + v_cfg * dt
                    nfe_total += 2 * current_batch_size

                # Corrector Step
                is_in_gate = args.tmin <= t_val <= args.tmax
                
                if args.method in ["cfg_mp_std", "cfg_mp_anderson"] or (args.method == "cfg_mp_anderson_gated" and is_in_gate):
                    t_next_val = (i + 1) / args.num_steps
                    t_next = torch.full((current_batch_size,), t_next_val, device=device)
                    drop_ids_proj = torch.ones(current_batch_size, dtype=torch.bool, device=device)

                    if args.method == "cfg_mp_std":
                        for k in range(1, args.proj_K):
                            v_proj = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj) 
                            v_proj, _ = v_proj.chunk(2, dim=1)
                            x = x + (v_proj - v_uncond_out) * dt * 0.5
                            nfe_total += current_batch_size

                    elif "anderson" in args.method:
                        v_proj_1 = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                        v_proj_1, _ = v_proj_1.chunk(2, dim=1)
                        g_1 = x + (v_proj_1 - v_uncond_out) * dt * 0.5
                        f_1 = g_1 - x

                        v_proj_2 = model(g_1, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                        v_proj_2, _ = v_proj_2.chunk(2, dim=1)
                        g_2 = g_1 + (v_proj_2 - v_uncond_out) * dt * 0.5
                        f_2 = g_2 - g_1

                        delta_f = f_2 - f_1
                        alpha = (torch.sum(f_2.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) / 
                                (torch.sum(delta_f.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) + 1e-8)).view(current_batch_size, 1, 1, 1)
                        
                        x = g_2 - alpha * (g_2 - g_1)
                        nfe_total += 2 * current_batch_size 

                z = x

            # Save FAKE images
            samples = torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
            for j in range(current_batch_size):
                save_image(samples[j], os.path.join(fake_dir, f"{total_generated + j:05d}.png"))
                
            total_generated += current_batch_size
            pbar.update(current_batch_size)

    # Save the exact conditions used in this run for the classifier eval
    torch.save(target_conditions, os.path.join(out_dir, "conditions.pt"))
    
    # Logging NFE Metrics
    avg_nfe = nfe_total / args.num_samples
    stats = {
        "method": args.method,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "total_nfe_batch": nfe_total,
        "avg_nfe_per_sample": avg_nfe
    }
    
    if "mp" in args.method:
        stats["proj_K"] = args.proj_K if args.method == "cfg_mp_std" else 2 
    if "gated" in args.method:
        stats["tmin"] = args.tmin
        stats["tmax"] = args.tmax

    log_path = os.path.join(out_dir, "generation_stats.json")
    with open(log_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nGeneration Complete! Saved to: {out_dir}")
    print(f"Average NFE per sample: {avg_nfe}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["uncond", "cfg", "cfg_mp_std", "cfg_mp_anderson", "cfg_mp_anderson_gated"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--attr-path", type=str, default="./data/local_celeba/attributes.pt", help="Path to the extracted attributes tensor")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--proj-K", type=int, default=3)
    parser.add_argument("--tmin", type=float, default=0.3)
    parser.add_argument("--tmax", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="DiT-B/2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    main(args)