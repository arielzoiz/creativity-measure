"""Decode Algorithm-3 k-sweep result latents into images. VAE only -- no transformer.

    python decode_run.py result_N8_K8_m1.25_seed101_k08b.pt [more.pt ...]

Writes <dir>/decoded/<tag>_p<j>.png plus one contact-sheet grid per checkpoint, and a combined
sheet when several checkpoints are given (one row per run, columns = particles).

Only the ~168 MB VAE is loaded, so this is quick and needs no staged transformer.

**Read the grid next to `uniq/M`.** A run that collapsed to one lineage produces eight columns that
are descendants of a single particle; they are not eight samples, however different they look.
"""
from __future__ import annotations

import os
import sys
from typing import Any, cast

import torch
from PIL import Image

CHUNK = 2


def load_vae(model_path: str, device: torch.device, dtype: torch.dtype) -> Any:
    """The VAE, on ``device``. Typed ``Any``: diffusers' stubs mistype ``.to`` and ``.config``."""
    from diffusers import AutoencoderKL

    vae = cast(Any, AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype))
    return vae.to(device).eval()


def main() -> None:
    paths = [p for p in sys.argv[1:] if not p.startswith("-")]
    if not paths:
        raise SystemExit(__doc__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = os.environ.get("FLUX_LOCAL_PATH") or "black-forest-labs/FLUX.1-dev"

    out_dir = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "decoded")
    os.makedirs(out_dir, exist_ok=True)
    print(f"model {model_path}\ndevice {device} ({dtype})\nout {out_dir}")

    vae = load_vae(model_path, device, dtype)
    from diffusers.image_processor import VaeImageProcessor

    proc = cast(Any, VaeImageProcessor(vae_scale_factor=8))
    sf = float(getattr(vae.config, "scaling_factor"))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    print(f"vae scaling_factor={sf}  shift_factor={shift}")

    rows: list[tuple[str, torch.Tensor]] = []
    for path in paths:
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = d["config"]
        C, H, W = cfg["base_process"]["img_shape"]
        X = d["X"]
        tag = (f"K{cfg['K']}_m{cfg['m_tilt']:g}_lam{cfg['lam']:.0f}"
               f"_uniq{d['uniq_history'][-1]:.3f}")
        outs = []
        for i in range(0, X.shape[0], CHUNK):
            lat = X[i:i + CHUNK].reshape(-1, C, H, W).to(device, dtype)
            lat = lat / sf + shift
            with torch.no_grad():
                sample = cast(torch.Tensor, vae.decode(lat).sample)
                outs.append(cast(torch.Tensor,
                                 proc.postprocess(sample.float(), output_type="pt")).cpu())
        imgs = torch.cat(outs)
        rows.append((tag, imgs))
        for j in range(imgs.shape[0]):
            arr = (imgs[j].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            Image.fromarray(arr).save(os.path.join(out_dir, f"{tag}_p{j}.png"))
        print(f"  {os.path.basename(path)} -> {tuple(imgs.shape)}  tag={tag}  "
              f"E_q[f]={d['eqf_history'][-1]:.4f}  uniq/M={d['uniq_history'][-1]:.3f}")

    # --- combined contact sheet: one row per run --------------------------------------------------
    n = min(r[1].shape[0] for r in rows)
    h, w = rows[0][1].shape[-2:]
    scale = 3                                   # thumbnails; the full-res PNGs are already written
    th, tw = h // scale, w // scale
    sheet = Image.new("RGB", (tw * n, th * len(rows)), "white")
    for r, (tag, imgs) in enumerate(rows):
        for j in range(n):
            arr = (imgs[j].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            sheet.paste(Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS), (tw * j, th * r))
    sheet_path = os.path.join(out_dir, "contact_sheet.png")
    sheet.save(sheet_path)
    print("\nrow order (top to bottom):")
    for tag, _ in rows:
        print(f"  {tag}")
    print(f"wrote {sheet_path}")


if __name__ == "__main__":
    main()
