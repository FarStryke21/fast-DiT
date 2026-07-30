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

    # --- UPDATED LOGIC: Use out_dir if provided, otherwise default ---
    if args.out_dir:
        out_dir = args.out_dir
    else:
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

    # Guidance-weight schedule. `constant` reproduces the original behaviour exactly.
    # `linear` is the mean-preserving increasing ramp of Wang et al. (arXiv:2404.13040):
    #   w(t) = 1 + (w - 1) * 2t   ->   w(0) = 1, w(1) = 2w - 1, mean over t in [0,1] = w.
    # Only applied to `cfg` and the `cfg_mp_*` family (see argparse help).
    def w_at(t_val):
        if args.w_schedule == "linear":
            return 1.0 + (args.cfg_scale - 1.0) * 2.0 * t_val
        return args.cfg_scale

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

            # Reusable force_drop_ids masks (no RNG, no per-step allocation semantics change)
            drop_none = torch.zeros(current_batch_size, dtype=torch.bool, device=device)
            drop_all = torch.ones(current_batch_size, dtype=torch.bool, device=device)

            def mp_icml_G(x_in, t_at):
                """CFG-MP fixed-point operator of arXiv:2601.21892 (Sec. 3.3):

                    G(x, t) = x - 0.5*dt*v_theta(t, x, None)
                                + 0.5*dt*v_theta(t, x - 0.5*dt*v_theta(t, x, None), y)

                Both velocities are evaluated at the SAME time `t_at` (the post-step time
                t_{i+1}). The two forwards are sequential (the conditional one depends on the
                unconditional one), so they cannot be batched: exactly 2 NFE per call.
                """
                v_u = model(x_in, t_at, y_uncond, force_drop_ids=drop_all)
                v_u, _ = v_u.chunk(2, dim=1)
                x_half = x_in - 0.5 * dt * v_u
                v_c = model(x_half, t_at, y_cond, force_drop_ids=drop_none)
                v_c, _ = v_c.chunk(2, dim=1)
                return x_half + 0.5 * dt * v_c

            for i in range(args.num_steps):
                t_val = i / args.num_steps
                t_next_val = (i + 1) / args.num_steps
                t = torch.full((current_batch_size,), t_val, device=device)
                w_t = w_at(t_val)

                # Predictor Step
                if args.method == "uncond":
                    # force_drop_ids selects the learned null_token; without it the model
                    # would be conditioned on the all-negative attribute vector instead
                    v = model(z, t, y_uncond, force_drop_ids=drop_all)
                    v, _ = v.chunk(2, dim=1)
                    x = z + v * dt
                    nfe_total += current_batch_size

                elif args.method == "cfg_interval":
                    # Guidance interval (Kynkaanniemi et al. 2024, arXiv:2404.07724): the guidance
                    # weight is applied only inside [w_tmin, w_tmax]; outside it the sampler uses
                    # the PURE CONDITIONAL velocity (equivalent to w = 1). Outside the interval the
                    # unconditional forward is therefore never needed -> 1 NFE instead of 2.
                    if args.w_tmin <= t_val <= args.w_tmax:
                        z_batched = torch.cat([z, z], dim=0)
                        t_batched = torch.cat([t, t], dim=0)
                        y_batched = torch.cat([y_cond, y_uncond], dim=0)
                        drop_ids = torch.cat([drop_none, drop_all])

                        v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
                        v_batched, _ = v_batched.chunk(2, dim=1)
                        v_cond, v_uncond_out = v_batched.chunk(2, dim=0)

                        v_step = v_uncond_out + args.cfg_scale * (v_cond - v_uncond_out)
                        nfe_total += 2 * current_batch_size
                    else:
                        v_step = model(z, t, y_cond, force_drop_ids=drop_none)
                        v_step, _ = v_step.chunk(2, dim=1)
                        nfe_total += current_batch_size
                    x = z + v_step * dt

                elif args.method == "cfg_pp":
                    # Flow-matching TRANSCRIPTION of CFG++ (Chung et al., arXiv:2406.08070) -- an
                    # ANALOGUE, not their exact algorithm. Their DDIM sampler denoises with the
                    # guided prediction but renoises with the UNCONDITIONAL one. Under the linear
                    # interpolant x_t = (1-t)x_0 + t*x_1 with v = x_1 - x_0 the implied endpoints
                    # are x1_hat = x_t + (1-t)v and x0_hat = x_t - t*v, so the same
                    # "denoise-guided, renoise-unconditional" rule reads:
                    #   x1_hat = x_t + (1-t)*v_guided ; x0_hat = x_t - t*v_uncond
                    #   x_{t+dt} = (1-(t+dt))*x0_hat + (t+dt)*x1_hat
                    # Here lambda = --cfg-scale lives in (0, 1], NOT a large CFG weight.
                    z_batched = torch.cat([z, z], dim=0)
                    t_batched = torch.cat([t, t], dim=0)
                    y_batched = torch.cat([y_cond, y_uncond], dim=0)
                    drop_ids = torch.cat([drop_none, drop_all])

                    v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
                    v_batched, _ = v_batched.chunk(2, dim=1)
                    v_cond, v_uncond_out = v_batched.chunk(2, dim=0)
                    nfe_total += 2 * current_batch_size

                    v_g = v_uncond_out + args.cfg_scale * (v_cond - v_uncond_out)
                    x1_hat = z + (1.0 - t_val) * v_g
                    x0_hat = z - t_val * v_uncond_out
                    x = (1.0 - t_next_val) * x0_hat + t_next_val * x1_hat

                else:
                    z_batched = torch.cat([z, z], dim=0)
                    t_batched = torch.cat([t, t], dim=0)
                    y_batched = torch.cat([y_cond, y_uncond], dim=0)
                    drop_ids = torch.cat([drop_none, drop_all])

                    v_batched = model(z_batched, t_batched, y_batched, force_drop_ids=drop_ids)
                    v_batched, _ = v_batched.chunk(2, dim=1)
                    v_cond, v_uncond_out = v_batched.chunk(2, dim=0)

                    v_cfg = v_uncond_out + w_t * (v_cond - v_uncond_out)
                    x = z + v_cfg * dt
                    nfe_total += 2 * current_batch_size

                # Corrector Step
                is_in_gate = args.tmin <= t_val <= args.tmax

                if args.method in ["cfg_mp_std", "cfg_mp_anderson"] or (args.method == "cfg_mp_anderson_gated" and is_in_gate):
                    t_next = torch.full((current_batch_size,), t_next_val, device=device)
                    drop_ids_proj = drop_all

                    # Anchor velocity v_bar for the fixed-point map G(x) = x + (v_(x,t') - v_bar)*dt*s.
                    # Default: reuse the (stale) v_uncond_out from the CFG pass at (z_t, t).
                    # --fresh-anchor: re-evaluate the unconditional field at t' at the UNCONDITIONAL
                    # Euler continuation x_uncond_next = z + v_uncond_out*dt -- i.e. where the
                    # trajectory would have gone with no guidance. Both sides of the fixed-point
                    # condition then live at t', killing the O(dt) stale-time bias. The anchor must
                    # NOT be the guided post-predictor point x: that point is a fixed point of its
                    # own corrector, which would make the whole corrector a no-op.
                    v_anchor = v_uncond_out
                    if args.fresh_anchor and (args.method != "cfg_mp_std" or args.proj_K > 1):
                        x_uncond_next = z + v_uncond_out * dt
                        v_anchor = model(x_uncond_next, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                        v_anchor, _ = v_anchor.chunk(2, dim=1)
                        nfe_total += current_batch_size

                    if args.method == "cfg_mp_std":
                        for k in range(1, args.proj_K):
                            v_proj = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                            v_proj, _ = v_proj.chunk(2, dim=1)
                            x = x + (v_proj - v_anchor) * dt * args.proj_step_scale
                            nfe_total += current_batch_size

                    elif "anderson" in args.method:
                        v_proj_1 = model(x, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                        v_proj_1, _ = v_proj_1.chunk(2, dim=1)
                        g_1 = x + (v_proj_1 - v_anchor) * dt * args.proj_step_scale
                        f_1 = g_1 - x

                        v_proj_2 = model(g_1, t_next, y_uncond, force_drop_ids=drop_ids_proj)
                        v_proj_2, _ = v_proj_2.chunk(2, dim=1)
                        g_2 = g_1 + (v_proj_2 - v_anchor) * dt * args.proj_step_scale
                        f_2 = g_2 - g_1

                        delta_f = f_2 - f_1
                        alpha = (torch.sum(f_2.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) /
                                (torch.sum(delta_f.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) + 1e-8)).view(current_batch_size, 1, 1, 1)
                        if args.alpha_clamp is not None:
                            alpha = alpha.clamp(-args.alpha_clamp, args.alpha_clamp)

                        x = g_2 - alpha * (g_2 - g_1)
                        nfe_total += 2 * current_batch_size

                elif args.method in ["cfg_mp_icml", "cfg_mp_icml_anderson"] or \
                        (args.method == "cfg_mp_icml_anderson_gated" and is_in_gate):
                    # Competitor corrector: arXiv:2601.21892, "Improving Classifier-Free Guidance of
                    # Flow Matching via Manifold Projection". Applied after the standard CFG Euler
                    # step, at the post-step time t' = t_{i+1} (verified against the paper's Sec. 3.3).
                    # cfg_mp_icml_anderson_gated applies our time gate to THEIR corrector: outside
                    # [tmin, tmax] this whole block is skipped, so those steps cost the CFG 2 NFE only.
                    t_next = torch.full((current_batch_size,), t_next_val, device=device)

                    if args.method == "cfg_mp_icml":
                        # Plain fixed-point iteration x <- G(x). We follow this repo's --proj-K
                        # convention (proj_K - 1 applications), so the default proj_K=3 gives 2
                        # iterations == the paper's recommended FPI = 2.
                        for k in range(1, args.proj_K):
                            x = mp_icml_G(x, t_next)
                            nfe_total += 2 * current_batch_size
                    else:
                        # Type-II Anderson, memory depth m=1, beta=1 -- identical extrapolation to
                        # our cfg_mp_anderson, only the operator G differs. The paper also uses
                        # AA(1,1) by default. Fixed cost: 2 applications of G (4 NFE) per corrected
                        # step -- for the _gated variant, only on steps inside [tmin, tmax].
                        g_1 = mp_icml_G(x, t_next)
                        f_1 = g_1 - x

                        g_2 = mp_icml_G(g_1, t_next)
                        f_2 = g_2 - g_1
                        nfe_total += 4 * current_batch_size

                        delta_f = f_2 - f_1
                        alpha = (torch.sum(f_2.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) /
                                (torch.sum(delta_f.view(current_batch_size, -1) * delta_f.view(current_batch_size, -1), dim=1) + 1e-8)).view(current_batch_size, 1, 1, 1)
                        if args.alpha_clamp is not None:
                            alpha = alpha.clamp(-args.alpha_clamp, args.alpha_clamp)

                        x = g_2 - alpha * (g_2 - g_1)

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
    
    if args.method.startswith("cfg_mp_icml"):
        # Applications of the arXiv:2601.21892 operator G per corrected step (2 NFE each);
        # for the _gated variant only steps with t in [tmin, tmax] are corrected at all.
        stats["mp_icml_iterations"] = max(args.proj_K - 1, 0) if args.method == "cfg_mp_icml" else 2
        if args.method == "cfg_mp_icml":
            stats["proj_K"] = args.proj_K
        if args.alpha_clamp is not None and "anderson" in args.method:
            stats["alpha_clamp"] = args.alpha_clamp
    elif "mp" in args.method:
        stats["proj_K"] = args.proj_K if args.method == "cfg_mp_std" else 2
        stats["proj_step_scale"] = args.proj_step_scale
        stats["fresh_anchor"] = args.fresh_anchor
        if args.alpha_clamp is not None:
            stats["alpha_clamp"] = args.alpha_clamp
    if args.method == "cfg" or args.method.startswith("cfg_mp"):
        stats["w_schedule"] = args.w_schedule
    if args.method == "cfg_interval":
        stats["w_tmin"] = args.w_tmin
        stats["w_tmax"] = args.w_tmax
    if args.method == "cfg_pp":
        # --cfg-scale is CFG++'s lambda in (0, 1] for this method, not a CFG weight w
        stats["cfg_pp_lambda"] = args.cfg_scale
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
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["uncond", "cfg", "cfg_interval", "cfg_pp",
                 "cfg_mp_std", "cfg_mp_anderson", "cfg_mp_anderson_gated",
                 "cfg_mp_icml", "cfg_mp_icml_anderson", "cfg_mp_icml_anderson_gated"],
        help=(
            "Sampler. "
            "uncond: unconditional baseline (1 NFE/step). "
            "cfg: vanilla velocity-space CFG (2 NFE/step). "
            "cfg_interval: guidance-interval baseline (Kynkaanniemi et al. 2024, arXiv:2404.07724) -- "
            "guidance weight --cfg-scale applies only for t in [--w-tmin, --w-tmax]; outside the "
            "interval the pure conditional velocity is used, so only the conditional forward runs "
            "(1 NFE/step outside, 2 inside). "
            "cfg_pp: our flow-matching analogue of CFG++ (Chung et al., arXiv:2406.08070) -- denoise "
            "with the guided velocity, renoise with the unconditional one; for THIS method --cfg-scale "
            "is lambda in (0, 1] (typical 0.2-0.8), NOT a large guidance weight (2 NFE/step). "
            "cfg_mp_std: our unconditional-anchor corrector, Picard iteration, proj_K-1 extra NFE/step. "
            "cfg_mp_anderson: same corrector with type-II Anderson (m=1), +2 NFE/step. "
            "cfg_mp_anderson_gated: cfg_mp_anderson fired only for t in [--tmin, --tmax]. "
            "cfg_mp_icml: competitor corrector of arXiv:2601.21892, "
            "G(x)=x-0.5*dt*v(t',x,null)+0.5*dt*v(t',x-0.5*dt*v(t',x,null),y) applied proj_K-1 times "
            "at the post-step time (+2 NFE per iteration). "
            "cfg_mp_icml_anderson: the same G wrapped in type-II Anderson (m=1), +4 NFE/step. "
            "cfg_mp_icml_anderson_gated: cfg_mp_icml_anderson fired only for t in [--tmin, --tmax] -- "
            "our time gate applied to the competitor's corrector (+4 NFE/step inside the gate, +0 outside)."
        ))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--attr-path", type=str, default="./data/local_celeba/attributes.pt", help="Path to the extracted attributes tensor")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--proj-K", type=int, default=3)
    parser.add_argument("--proj-step-scale", type=float, default=0.5, help="Corrector step size as a fraction of dt (previously hard-coded 0.5)")
    parser.add_argument("--alpha-clamp", type=float, default=None, help="Clamp Anderson mixing coefficient to [-c, c]; None keeps the original unclamped behaviour")
    parser.add_argument("--tmin", type=float, default=0.3)
    parser.add_argument("--tmax", type=float, default=0.7)
    parser.add_argument("--w-tmin", type=float, default=0.3,
                        help="cfg_interval only: start of the guidance interval (t=0 is noise)")
    parser.add_argument("--w-tmax", type=float, default=0.7,
                        help="cfg_interval only: end of the guidance interval")
    parser.add_argument("--w-schedule", type=str, choices=["constant", "linear"], default="constant",
                        help=("Guidance-weight schedule for `cfg` and all `cfg_mp_*` methods. "
                              "constant: w(t) = --cfg-scale. "
                              "linear: mean-preserving increasing ramp w(t) = 1 + (w-1)*2t "
                              "(Wang et al., arXiv:2404.13040) -- starts at 1, ends at 2w-1, "
                              "averages w over t in [0,1]."))
    parser.add_argument("--fresh-anchor", action="store_true",
                        help=("cfg_mp_std / cfg_mp_anderson / cfg_mp_anderson_gated only: recompute the "
                              "corrector anchor as v_bar = v(x_uncond_next, t_next, null), where "
                              "x_uncond_next = z + v_uncond*dt is the unconditional Euler continuation, "
                              "instead of reusing the stale v_uncond from the CFG pass at (z_t, t). "
                              "Costs +1 NFE per corrected step."))
    parser.add_argument("--model", type=str, default="DiT-B/2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=50)
    # --- UPDATED: Added out-dir argument ---
    parser.add_argument("--out-dir", type=str, default=None, help="Force a specific output directory")
    args = parser.parse_args()
    main(args)