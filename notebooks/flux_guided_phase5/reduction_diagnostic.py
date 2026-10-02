"""Why did preflight's bitwise reduction check fail on GPU when it passes on CPU?

Jobs 965868/965869 died at `preflight reduction: ... bitwise=False (max abs diff 1.174e+01)`, i.e.
`flow_guided_pc_sample(corrector_steps=0)` did NOT reproduce `flow_guided_sample` on real FLUX at
lam=1.0, exact_jacobian=True -- although tests/test_flow_guided_pc.py asserts exactly that, bitwise, on
CPU against a tiny transformer, and it passes there in both Jacobian modes.

Two hypotheses, and they demand opposite responses:

  (H1) MODEL NONDETERMINISM. The backward pass through FLUX is not bit-reproducible on GPU -- flash
       attention's backward uses atomics, and `_freeze` enables gradient checkpointing, so the backward
       RECOMPUTES the forward. Then no two gradient evaluations agree bitwise, the two samplers get
       slightly different `grad r`, and with guidance renormalized to ||v_theta|| the difference is
       amplified into an O(1) trajectory divergence within a couple of steps. If so, my assert was
       simply the wrong assert for exact_jacobian=True, nothing is broken, and the fix is in the check.

  (H2) A REAL DIFFERENCE between the two code paths that the CPU test cannot see. If so, loosening the
       check would launder a genuine bug into 20+ GPU-hours of meaningless numbers.

THE DECISIVE TEST is whether `flow_guided_sample` can reproduce ITSELF. It is one function, called twice
with the same seed; any disagreement is nondeterminism by definition and cannot be a difference between
samplers. If self-disagreement is the same order as the cross-sampler disagreement, H1 is established
and H2 is excluded.

Note what Phase 3 actually verified on hardware: job 957386 reported "lam=0 bitwise parity holds on real
hardware too" -- at lam=0, where the guided branch never runs and NO backward is taken. Nobody has ever
checked bitwise reproducibility of the lam!=0, exact_jacobian=True path on a GPU. This script does.

    python reduction_diagnostic.py            # GPU
    python reduction_diagnostic.py --dry-run  # CPU, tiny model (expect ALL bitwise)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flux_guided_phase3"))

from fine_lambda_sweep import (   # noqa: E402  # pyright: ignore[reportMissingImports]
    N_PARTICLES, SHIFT, Setup, dry_setup, flux_setup, note,
)

from creativity_measure.samplers.flow_guided import flow_guided_sample          # noqa: E402
from creativity_measure.samplers.flow_guided_pc import flow_guided_pc_sample    # noqa: E402

N_STEPS = 2
OUT = os.path.join(HERE, "reduction_diagnostic.json")


def _run(S: Setup, which: str, lam: float, exact: bool, seed: int) -> torch.Tensor:
    kw: dict[str, Any] = dict(
        velocity_fn=S.velocity_fn, n_steps=N_STEPS, shift=SHIFT, t_start=1.0, t_end=0.0,
        exact_jacobian=exact, seed=seed,
    )
    if which == "fg":
        return flow_guided_sample(S.reward, lam, N_PARTICLES, **kw).X
    return flow_guided_pc_sample(S.reward, lam, N_PARTICLES, corrector_steps=0, **kw).X


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = (a.float() - b.float()).abs()
    return {
        "bitwise": bool(torch.equal(a, b)),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rel": float(d.norm() / a.float().norm().clamp_min(1e-30)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    S = dry_setup() if args.dry_run else flux_setup()
    note(f"setup ready: d={S.d} device={S.device} lambda_s={S.lam_s:.3f}")

    cases = [
        # (label, what it decides)
        ("self_fg_exact",    "flow_guided vs ITSELF, lam=1, exact=True  <- THE decisive test"),
        ("self_fg_approx",   "flow_guided vs ITSELF, lam=1, exact=False <- is the approx path stable?"),
        ("self_fg_lam0",     "flow_guided vs ITSELF, lam=0              <- is the no-backward path stable?"),
        ("cross_lam0",       "flow_guided vs pc(corr=0), lam=0          <- what Phase 3 verified"),
        ("cross_approx",     "flow_guided vs pc(corr=0), lam=1, exact=False"),
        ("cross_exact",      "flow_guided vs pc(corr=0), lam=1, exact=True  <- what preflight asserted"),
    ]
    out: dict[str, Any] = {"d": S.d, "n_steps": N_STEPS, "dry_run": args.dry_run,
                           "gpu": (None if args.dry_run else torch.cuda.get_device_name(0))}

    for label, desc in cases:
        t0 = time.time()
        if label == "self_fg_exact":
            a, b = _run(S, "fg", 1.0, True, 7), _run(S, "fg", 1.0, True, 7)
        elif label == "self_fg_approx":
            a, b = _run(S, "fg", 1.0, False, 7), _run(S, "fg", 1.0, False, 7)
        elif label == "self_fg_lam0":
            a, b = _run(S, "fg", 0.0, True, 7), _run(S, "fg", 0.0, True, 7)
        elif label == "cross_lam0":
            a, b = _run(S, "fg", 0.0, True, 7), _run(S, "pc", 0.0, True, 7)
        elif label == "cross_approx":
            a, b = _run(S, "fg", 1.0, False, 7), _run(S, "pc", 1.0, False, 7)
        else:
            a, b = _run(S, "fg", 1.0, True, 7), _run(S, "pc", 1.0, True, 7)
        r = _cmp(a, b)
        r["desc"] = desc
        r["elapsed_s"] = time.time() - t0
        out[label] = r
        note(f"{label:16s} bitwise={str(r['bitwise']):5s} max_abs={r['max_abs']:.4e} "
             f"rel={r['rel']:.4e}  ({r['elapsed_s']:.0f}s)  -- {desc}")
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)

    print("\n" + "=" * 100)
    se = out["self_fg_exact"]["max_abs"]
    ce = out["cross_exact"]["max_abs"]
    if out["self_fg_exact"]["bitwise"]:
        print("VERDICT: flow_guided_sample IS self-reproducible at exact_jacobian=True, so the")
        print("cross-sampler disagreement is NOT nondeterminism -- hypothesis H2, a real difference")
        print("between the paths. DO NOT loosen the preflight check; find the bug.")
    elif se > 0 and 0.1 <= ce / se <= 10.0:
        print("VERDICT: flow_guided_sample cannot reproduce ITSELF at exact_jacobian=True")
        print(f"(self max_abs {se:.4e} vs cross {ce:.4e}, same order) -- hypothesis H1, FLUX's backward")
        print("is nondeterministic on GPU (flash-attention atomics + gradient-checkpoint recompute).")
        print("Nothing is broken; a BITWISE assert is simply invalid for the exact-Jacobian path.")
    else:
        print(f"VERDICT: AMBIGUOUS. self max_abs {se:.4e} vs cross {ce:.4e} differ by more than 10x;")
        print("nondeterminism is present but may not account for the whole cross-sampler gap.")
    print("=" * 100)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
