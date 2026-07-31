"""Measure the real denoiser batch-width ceiling for FLUX.1-dev on this card.

THE QUESTION. After commit e1bfc99 the IEM reference score bank is built by ONE call per gamma
interval on ``NUM_EPS * R`` rows (``global_iem._scores_at_gamma``), so R is now a *batch dimension*
rather than a loop counter. Run 2 wants R=64, NUM_EPS=3 -> 192 rows. Job 694278 (which worked) never
exceeded 24 rows, and job 694256 is recorded as dying with CUDA OOM at exactly 192. This script
replaces that inference with a measurement.

WHAT IT DOES. Loads the transformer + prompt embeddings the sweep uses, rebuilds the identical
denoiser closure, then runs one forward pass at each candidate width, recording peak VRAM and
seconds/row. Widths are tried ASCENDING and each phase stops at its first OOM, so the largest
surviving width is the answer.

TWO PHASES, ONE MODEL LOAD. Phase A probes an otherwise-empty card. Phase B allocates the resident
reference score bank -- (N_GAMMA-1, NUM_EPS, R, d) fp32, 1.36 GB at R=64/N_GAMMA=30 -- which the real
run holds while issuing these forwards, and probes again. The difference isolates what the bank costs
in headroom. Job 695141 reloaded the model per phase and never got to probe at all; loading once is
the fix.

WATCHDOG. Armed only around the probe loops. Job 695141 died with a 480 s alarm firing during
``.to(DEVICE)``, which takes ~7 min on this cluster (the checkpoint shards themselves load in 9 s).
slurm ``--time`` is the backstop for the load phase.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", type=int, nargs="+", default=[24, 48, 96, 128, 192, 256],
                   help="denoiser batch widths to try, ascending")
    p.add_argument("--n-gamma", type=int, default=30)
    p.add_argument("--num-eps", type=int, default=3)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--img", type=int, default=512)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--prompt", default="A dog")
    p.add_argument("--timeout", type=int, default=900,
                   help="watchdog on the PROBE LOOPS only; the model load is not covered")
    p.add_argument("--out", default="probe_denoiser_batch.json")
    return p.parse_args()


class Watchdog(RuntimeError):
    pass


def main() -> None:
    args = parse_args()
    t0_all = time.time()

    def stamp(msg: str) -> None:
        print(f"[{time.time() - t0_all:6.0f}s] {msg}", flush=True)

    def _bark(signum, frame):
        raise Watchdog(f"watchdog fired after {args.timeout}s in the probe loop")

    signal.signal(signal.SIGALRM, _bark)

    import torch
    from diffusers import FluxPipeline

    assert torch.cuda.is_available(), "no CUDA device"
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    total_gb = torch.cuda.mem_get_info()[1] / 2**30
    stamp(f"GPU: {torch.cuda.get_device_name(0)}  {total_gb:.1f} GB total")

    IMG = args.img
    C, H, W = 16, IMG // 8, IMG // 8
    LATENT_DIM = C * H * W

    # --- load (NOT under the watchdog; slurm --time is the backstop) --------------------------------
    stamp("from_pretrained ...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE)
    stamp("  -> loaded to CPU;  .to(cuda) ...   (this is the slow step, ~7 min observed)")
    pipe = pipe.to(DEVICE)
    stamp(f"  -> on GPU, {torch.cuda.memory_allocated() / 2**30:.2f} GB resident;  encode_prompt ...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=args.prompt, prompt_2=None, device=DEVICE, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    torch.cuda.empty_cache()
    transformer = pipe.transformer.eval()

    img_ids = FluxPipeline._prepare_latent_image_ids(1, H // 2, W // 2, DEVICE, DTYPE)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    def denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Byte-for-byte the sweep notebook's closure (cell 3)."""
        b = x.shape[0]
        sig = sigma.reshape(b, 1, 1, 1).to(torch.float32)
        t = (sigma / (1.0 + sigma)).to(torch.float32)
        x_t = x.to(torch.float32) / (1.0 + sig)
        packed = FluxPipeline._pack_latents(x_t.to(DTYPE), b, C, H, W)
        with torch.no_grad():
            v = transformer(
                hidden_states=packed,
                timestep=t.to(DTYPE),
                guidance=torch.full((b,), args.guidance, device=x.device, dtype=torch.float32),
                pooled_projections=pooled_prompt_embeds.expand(b, -1).to(DTYPE),
                encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(DTYPE),
                txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
        v = FluxPipeline._unpack_latents(v, IMG, IMG, 8).to(torch.float32)
        return (x_t - t.reshape(b, 1, 1, 1) * v).to(torch.float32)

    model_gb = torch.cuda.memory_allocated() / 2**30
    t_load = time.time() - t0_all
    stamp(f"MODEL READY in {t_load:.0f}s   resident {model_gb:.2f} GB   "
          f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")

    bank_gb = (args.n_gamma - 1) * args.num_eps * args.R * LATENT_DIM * 4 / 2**30

    def probe_widths(label: str) -> list[dict]:
        out = []
        stamp(f"--- {label} ---")
        print(f"{'rows':>6} {'status':>8} {'peak GB':>9} {'free GB':>9} {'sec':>7} {'s/row':>8}",
              flush=True)
        for wdt in sorted(args.widths):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            rec: dict = {"rows": wdt, "phase": label}
            try:
                x = torch.randn(wdt, C, H, W, device=DEVICE, dtype=torch.float32)
                sig = torch.full((wdt,), 1.0, device=DEVICE, dtype=torch.float32)
                torch.cuda.synchronize()
                t0 = time.time()
                res = denoiser(x, sig)
                torch.cuda.synchronize()
                dt = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 2**30
                free = torch.cuda.mem_get_info()[0] / 2**30
                rec.update(status="ok", peak_gb=round(peak, 3), free_gb=round(free, 3),
                           sec=round(dt, 2), s_per_row=round(dt / wdt, 4),
                           out_shape=list(res.shape))
                print(f"{wdt:6d} {'ok':>8} {peak:9.2f} {free:9.2f} {dt:7.2f} {dt / wdt:8.4f}",
                      flush=True)
                del x, sig, res
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                        and "out of memory" not in str(exc).lower():
                    raise
                rec.update(status="OOM", error=str(exc)[:200])
                print(f"{wdt:6d} {'OOM':>8}   <-- ceiling is below this width", flush=True)
                torch.cuda.empty_cache()
                out.append(rec)
                break
            out.append(rec)
        return out

    results: list[dict] = []
    signal.alarm(args.timeout)
    try:
        results += probe_widths("PHASE A: empty card")

        stamp(f"allocating resident reference bank "
              f"(N_GAMMA-1={args.n_gamma - 1}, NUM_EPS={args.num_eps}, R={args.R}, d={LATENT_DIM}) "
              f"fp32 = {bank_gb:.2f} GB")
        bank = torch.empty((args.n_gamma - 1, args.num_eps, args.R, LATENT_DIM),
                           device=DEVICE, dtype=torch.float32)
        stamp(f"  free after bank: {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")

        results += probe_widths("PHASE B: holding the reference bank (the real condition)")
        del bank
    finally:
        signal.alarm(0)

    def summarise(phase: str) -> dict:
        rows = [r for r in results if r["phase"].startswith(phase)]
        ok = [r["rows"] for r in rows if r["status"] == "ok"]
        oom = [r["rows"] for r in rows if r["status"] == "OOM"]
        return {"largest_ok": max(ok) if ok else None, "smallest_oom": min(oom) if oom else None}

    a, b = summarise("PHASE A"), summarise("PHASE B")
    needed_rows = args.num_eps * args.R
    fits = b["largest_ok"] is not None and b["largest_ok"] >= needed_rows
    verdict = {
        "phase_a_empty_card": a, "phase_b_with_bank": b,
        "rows_needed_for_ref_bank": needed_rows,
        "chunking_required": not fits,
        "bank_gb": round(bank_gb, 3), "model_gb": round(model_gb, 2),
        "gpu": torch.cuda.get_device_name(0), "total_gb": round(total_gb, 1),
        "model_load_s": round(t_load, 1),
    }
    print("\n" + "=" * 72)
    print(f"PHASE A (empty card)   largest ok: {a['largest_ok']}   first OOM: {a['smallest_oom']}")
    print(f"PHASE B (bank held)    largest ok: {b['largest_ok']}   first OOM: {b['smallest_oom']}")
    print(f"\nrun 2 needs NUM_EPS*R = {needed_rows} rows for the reference-bank build")
    print("VERDICT: CHUNKING REQUIRED" if not fits else
          f"VERDICT: NO CHUNKING NEEDED -- {needed_rows} rows fit with the bank resident")
    print("=" * 72)

    with open(args.out, "w") as fh:
        json.dump({"verdict": verdict, "results": results, "args": vars(args)}, fh, indent=2)
    stamp(f"wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Watchdog as exc:
        print(f"\n*** {exc} -- aborting; measurements printed above are still valid",
              file=sys.stderr)
        sys.exit(2)
