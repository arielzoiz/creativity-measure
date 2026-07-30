"""Re-decode saved per-level latents into images, using only the VAE.

The λ-sweep notebook saves each tempering level's particle cloud (``X``) before decoding. This script
turns any of those latents back into images **without loading the 24 GB FLUX transformer** -- only the
~168 MB VAE is needed, so it runs on a small GPU or on CPU. That makes re-rendering a finished run
(different levels, different layout, higher-quality output) essentially free.

This script lives alongside the notebook and the checkpoints it reads, so from this directory the
checkpoint name is all that is needed. Nothing here imports ``creativity_measure`` -- only torch,
diffusers and PIL -- so it runs from anywhere without the repo on ``sys.path``.

Usage
-----
    cd notebooks/flux_lambda_sweep
    python decode_levels.py levels_N8_lam123_seed101.pt              # all levels, GPU if present
    python decode_levels.py levels_N8_lam123_seed101.pt --levels 0 3 -1
    python decode_levels.py levels_N8_lam123_seed101.pt --device cpu --no-grid   # PNGs only

Output goes to ``<checkpoint dir>/decoded/`` unless ``--out-dir`` says otherwise.

Grid rows are λ_eff ascending; columns follow lineage (walking ``ancestors`` backwards from the final
level) so a column tracks one particle as the tilt strengthens -- same convention as the notebook.
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="the .pt written by the notebook's run cell")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="level indices to decode (negatives allowed); default: all")
    p.add_argument("--out-dir", default=None,
                   help="where to write PNGs and the grid; default: alongside the checkpoint")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--chunk", type=int, default=2, help="VAE decode batch size (memory spike)")
    p.add_argument("--no-png", action="store_true", help="skip per-image PNGs")
    p.add_argument("--no-grid", action="store_true", help="skip the lambda_eff x particle grid")
    p.add_argument("--lineage", action="store_true", default=True,
                   help="order columns by lineage (default)")
    p.add_argument("--no-lineage", dest="lineage", action="store_false",
                   help="keep raw particle order instead")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # fp32 on CPU: bf16 VAE decode on CPU is slow and can be numerically poor.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    levels = ckpt["levels"]
    n = levels[0]["X"].shape[0]
    C, H, W = cfg["C"], cfg["H"], cfg["W"]

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "decoded")
    os.makedirs(out_dir, exist_ok=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"model      : {cfg['model_id']}")
    print(f"levels     : {len(levels)}  particles: {n}  latent: ({C}, {H}, {W})")
    print(f"prompt     : {cfg['prompt']!r}   lam_max={cfg['lam_max']:.1f}")
    print(f"device     : {device} ({dtype})")
    print(f"out dir    : {out_dir}")

    # --- VAE only: no transformer, no text encoders ---------------------------------------------
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKL.from_pretrained(
        cfg["model_id"], subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    proc = VaeImageProcessor(vae_scale_factor=8)
    sf, shift = cfg["vae_scaling_factor"], cfg["vae_shift_factor"]

    def decode(flat: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, flat.shape[0], args.chunk):
            lat = flat[i : i + args.chunk].reshape(-1, C, H, W).to(device, dtype)
            lat = lat / sf + shift
            with torch.no_grad():
                img = vae.decode(lat).sample
            outs.append(proc.postprocess(img.float(), output_type="pt").cpu())
        return torch.cat(outs)

    # --- lineage column order (same walk as the notebook) ---------------------------------------
    col_of = [torch.arange(n)] * len(levels)
    if args.lineage:
        col_of = [None] * len(levels)
        cur = torch.arange(n)
        col_of[-1] = cur
        for k in range(len(levels) - 1, 0, -1):
            cur = levels[k]["ancestors"][cur]
            col_of[k - 1] = cur

    wanted = args.levels if args.levels is not None else list(range(len(levels)))
    wanted = [k % len(levels) for k in wanted]      # allow -1 for the final level

    decoded: dict[int, torch.Tensor] = {}
    for k in wanted:
        imgs = decode(levels[k]["X"][col_of[k]])
        decoded[k] = imgs
        lam_eff = levels[k]["lam_eff"]
        print(f"  level {k:2d}  lam_eff={lam_eff:9.1f}  -> {tuple(imgs.shape)}")
        if not args.no_png:
            # PIL rather than torchvision.utils.save_image: torchvision is not a dependency of
            # this project and is not installed in the creativity-measure env.
            for j in range(imgs.shape[0]):
                arr = (imgs[j].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                Image.fromarray(arr).save(
                    os.path.join(out_dir, f"lvl{k:02d}_lam{lam_eff:.0f}_p{j}.png")
                )

    if args.no_grid:
        print("done (grid skipped)")
        return

    import matplotlib
    matplotlib.use("Agg")           # headless: this may run inside a batch job
    import matplotlib.pyplot as plt

    ks = sorted(decoded)
    nrow, ncol = len(ks), n
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.05), squeeze=False)
    for r, k in enumerate(ks):
        for j in range(ncol):
            ax = axes[r][j]
            ax.axis("off")
            ax.imshow(decoded[k][j].permute(1, 2, 0).clamp(0, 1).numpy())
            if j == 0:
                ax.text(-0.12, 0.5,
                        f"$\\lambda_{{eff}}$={levels[k]['lam_eff']:.0f}\n"
                        f"$\\beta$={levels[k]['beta']:.3f}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=8)
            if r == 0:
                ax.set_title(f"col {j}", fontsize=8)
    order = "lineage" if args.lineage else "raw particle index"
    fig.suptitle(
        f"$\\lambda$ sweep from one SMC run  (prompt: {cfg['prompt']!r}, N={n}, "
        f"$\\lambda_{{max}}$={cfg['lam_max']:.0f})\nrows: $\\lambda_{{eff}}$ ascending   "
        f"columns: {order}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0.03, 0, 1, 0.97))
    grid_path = os.path.join(out_dir, "lambda_sweep_grid.png")
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
