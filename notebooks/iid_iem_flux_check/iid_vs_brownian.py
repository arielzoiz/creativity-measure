"""GPU check for the i.i.d. Monte-Carlo squared-IEM reward (Phase 1), on FLUX.1-dev.

The CPU tests (tests/test_iid_global_iem.py) prove the math on a 2D toy. They cannot say whether a FROZEN bank of
G ~ 30-50 log-uniform gammas is accurate enough on FLUX's d = 65536 latents, nor that FLUX's transformer handles a
per-row noise level. Two stages, each appending to `iid_vs_brownian_results.json` as it goes:

  G0  smoke (PASS/FAIL)   `edm_score_fn` with one gamma PER ROW == one call per gamma, on the real transformer.
                          Includes a negative control (wrong gamma on every row) so the check can actually fail.
  G1  agreement (numbers) f_IID vs f_Brownian on the same 32 held-out latents and the same R = 64 references.
                          The yardstick is Brownian vs ITSELF under a different bank seed (123 vs 124): f has a CV of
                          ~0.7% across latents (CLAUDE.md), so ranks are fragile and "how well do two Brownian banks
                          agree" is the honest ceiling for any estimator of this f.

Everything is the production reward: `NormalizedExpectedDistanceReward`, fp32, the same model / prompt / guidance /
gamma window / denoiser chunking as `notebooks/refset_auto_r/auto_r_common.py` (imported, not copied).

    python iid_vs_brownian.py            # on a GPU node (see iid_vs_brownian.slurm)
    python iid_vs_brownian.py --dry-run  # CPU, tiny GMM stand-in for FLUX: exercises all the plumbing, no GPU needed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "refset_auto_r"))

from creativity_measure import (                                   # noqa: E402
    NormalizedExpectedDistanceReward, SquaredGlobalIEMDistance, SquaredIIDGlobalIEMDistance,
    edm_score_fn, log_uniform_gammas,
)

REF_SEED, HELDOUT_SEED = 7, 8               # production's (flowmap_smc_flux.ipynb: REF_SEED, DIST_SEED, HELDOUT_SEED)
R_REFS, N_HELDOUT = 64, 32
BROWNIAN_SEEDS = (123, 124)                 # 123 is production's; 124 is the yardstick's second bank
IID_CONFIGS = ((30, 1), (50, 1), (30, 2))   # (G, N_eps)
IID_SEEDS = (0, 1, 2)                       # each seeds BOTH the gamma draw and the eps bank
RESULTS = os.path.join(HERE, "iid_vs_brownian_results.json")
S_ROW = 0.22                                # s per transformer row, from auto_r_common's measurement


class Sampler(Protocol):
    def sample(self, n: int, seed: int | None = None) -> Tensor: ...


@dataclass
class Setup:
    device: torch.device
    p_gen: Sampler                          # .sample(n, seed=) -> (n, d)
    score_fn: Callable[[Tensor, Tensor], Tensor]
    gammas: Tensor                          # the production logspace grid; only its endpoints are used here
    n_gamma: int
    num_eps: int
    stamp: dict


# =====================================================================================================
# Setups: the real one (FLUX) and a CPU stand-in that exercises the identical code path
# =====================================================================================================

def flux_setup() -> Setup:
    import auto_r_common as arc                                    # pyright: ignore[reportMissingImports]  (path added above)
    s = arc.build()
    stamp = {"model": arc.MODEL_ID, "prompt": arc.PROMPT, "guidance": arc.GUIDANCE, "img": arc.IMG,
             "n_gamma": arc.N_GAMMA, "num_eps": arc.NUM_EPS, "gpu": torch.cuda.get_device_name(0)}
    return Setup(s.device, s.p_gen, s.distance.score_fn, s.gammas, arc.N_GAMMA, arc.NUM_EPS, stamp)


def dry_setup() -> Setup:
    """A 2-mode Gaussian mixture in d = 64 with its exact per-row-sigma denoiser: no GPU, no FLUX, same code path."""
    d, m, s2 = 64, 0.4, 0.05
    mu = torch.stack([torch.full((d,), m), torch.full((d,), -m)])           # (2, d)

    def denoiser(y_sigma: Tensor, sigma: Tensor) -> Tensor:                # E[X | y_sigma], sigma is (B,)
        v = s2 + sigma.reshape(-1, 1) ** 2                                  # (B, 1)
        logit = -((y_sigma.unsqueeze(1) - mu.unsqueeze(0)) ** 2).sum(-1) / (2 * v)   # (B, 2)
        r = torch.softmax(logit, dim=1).unsqueeze(-1)                       # (B, 2, 1)
        post = mu.unsqueeze(0) + (s2 / v).unsqueeze(-1) * (y_sigma.unsqueeze(1) - mu.unsqueeze(0))
        return (r * post).sum(1)

    class Sampler:
        def sample(self, n: int, seed: int | None = None) -> Tensor:
            g = torch.Generator().manual_seed(0 if seed is None else int(seed))
            k = torch.randint(0, 2, (n,), generator=g)
            return mu[k] + math.sqrt(s2) * torch.randn(n, d, generator=g)

    gammas = torch.logspace(-6, 6, 30, base=2)
    return Setup(torch.device("cpu"), Sampler(), edm_score_fn(denoiser), gammas, 30, 3, {"dry_run": True, "d": d})


# =====================================================================================================
# Results file (atomic, resumable per config)
# =====================================================================================================

def load_results(stamp: dict) -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            old = json.load(fh)
        if old.get("stamp") == stamp:
            print(f"[resume] {RESULTS}: {sorted(k for k in old if k not in ('stamp',))}")
            return old
        os.replace(RESULTS, RESULTS + ".stale")
        print("[resume] stamp mismatch -> old results moved to .stale, starting fresh")
    return {"stamp": stamp}


def save_results(res: dict) -> None:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2)
    os.replace(tmp, RESULTS)


def note(msg: str) -> None:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform != "darwin" else 1024 ** 3)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (host peak RSS {rss:.1f} GB)", flush=True)


# =====================================================================================================
# G0 -- per-row gamma on the real transformer
# =====================================================================================================

def run_g0(S: Setup, x_pool: Tensor, res: dict) -> None:
    if "g0" in res:
        note(f"G0 already done: {res['g0']['verdict']}")
        return
    lo, hi = float(S.gammas[0]), float(S.gammas[-1])
    n = 4
    g = torch.logspace(math.log10(lo) + 0.5, math.log10(hi) - 0.5, n).to(S.device, x_pool.dtype)   # spans the window
    x = x_pool[:n]
    gen = torch.Generator(device=S.device).manual_seed(0)
    eps = torch.randn(n, x.shape[1], device=S.device, dtype=x.dtype, generator=gen)
    y = g.view(-1, 1) * x + g.sqrt().view(-1, 1) * eps

    with torch.no_grad():
        fused = S.score_fn(y, g)                                                         # one call, gamma per row
        looped = torch.cat([S.score_fn(y[i:i + 1], g[i]) for i in range(n)], dim=0)      # one call per gamma
        control = S.score_fn(y, g[0])                                                    # WRONG: row-0 gamma on all rows

    # Compare on the DENOISER's scale, ||y/gamma||: score = D - y/gamma, so score differences ARE denoiser differences,
    # and dividing by ||score|| instead would blow up at large gamma where D - y/gamma is a small difference of big terms.
    scale = (y / g.view(-1, 1)).norm(dim=1)
    err_fused = ((fused - looped).norm(dim=1) / scale)
    err_ctrl = ((control - looped).norm(dim=1) / scale)
    ok = bool(err_fused.max() < 5e-2) and bool(err_fused.max() < 0.25 * err_ctrl[1:].max())
    res["g0"] = {"verdict": "PASS" if ok else "FAIL", "gammas": g.tolist(),
                 "rel_err_fused_vs_looped": err_fused.tolist(), "rel_err_negative_control": err_ctrl.tolist(),
                 "rule": "max fused err < 5e-2 (bf16) AND < 0.25 x the negative control's max (rows 1..3)"}
    save_results(res)
    note(f"G0 {res['g0']['verdict']}: fused-vs-looped rel err {err_fused.max():.2e} "
         f"(negative control {err_ctrl[1:].max():.2e})")


# =====================================================================================================
# G1 -- f_IID vs f_Brownian
# =====================================================================================================

def eval_f(dist, refs: Tensor, X: Tensor) -> tuple[np.ndarray, float]:
    """Production reward: f = E_refs[D^2](x) / E_pairs[D^2], normalizer built from the same distance and refs."""
    reward = NormalizedExpectedDistanceReward(dist, refs)
    with torch.no_grad():
        f = reward(X)
    return f.double().cpu().numpy(), float(reward._denom)


def run_brownian(S: Setup, refs: Tensor, X: Tensor, seed: int, res: dict) -> None:
    key = f"brownian_{seed}"
    if key in res:
        note(f"{key} already done")
        return
    t0 = time.time()
    dist = SquaredGlobalIEMDistance(None, S.gammas, num_eps=S.num_eps, seed=seed, score_fn=S.score_fn)
    f, denom = eval_f(dist, refs, X)
    rows = (S.n_gamma - 1) * S.num_eps * (refs.shape[0] + X.shape[0])
    res[key] = {"f": f.tolist(), "denom": denom, "seconds": time.time() - t0, "score_rows": rows}
    save_results(res)
    note(f"{key}: f mean {f.mean():.5f} sd {f.std():.5f}  denom {denom:.4g}  {time.time() - t0:.0f}s ({rows} rows)")
    del dist
    _free()


def run_iid(S: Setup, refs: Tensor, X: Tensor, G: int, E: int, seed: int, res: dict) -> None:
    key = f"iid_G{G}_E{E}_s{seed}"
    if key in res:
        note(f"{key} already done")
        return
    t0 = time.time()
    lo, hi = float(S.gammas[0]), float(S.gammas[-1])
    g, w = log_uniform_gammas(lo, hi, G, seed, device=S.device, dtype=X.dtype)
    dist = SquaredIIDGlobalIEMDistance(None, g, w, num_eps=E, seed=seed, score_fn=S.score_fn)
    f, denom = eval_f(dist, refs, X)
    rows = G * E * (refs.shape[0] + X.shape[0])
    res[key] = {"f": f.tolist(), "denom": denom, "seconds": time.time() - t0, "score_rows": rows}
    save_results(res)
    note(f"{key}: f mean {f.mean():.5f} sd {f.std():.5f}  denom {denom:.4g}  {time.time() - t0:.0f}s ({rows} rows)")
    del dist
    _free()


def _free() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Rank and linear agreement of two f vectors over the same latents (Spearman = Pearson of average ranks)."""
    return {"spearman": float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), "pearson": float(np.corrcoef(a, b)[0, 1])}


def summarize(res: dict) -> dict:
    """Table of every IID config against Brownian(123), with Brownian(124)-vs-(123) as the yardstick."""
    b1, b2 = (np.array(res[f"brownian_{s}"]["f"]) for s in BROWNIAN_SEEDS)
    sd_ref = float(b1.std(ddof=1))
    out: dict = {"yardstick_brownian_124_vs_123": agreement(b2, b1)}
    out["yardstick_brownian_124_vs_123"]["mean_abs_diff_over_sd"] = float(np.abs(b2 - b1).mean() / sd_ref)
    out["n_heldout"] = int(b1.shape[0])
    out["brownian_score_rows_per_reward_build"] = int(res[f"brownian_{BROWNIAN_SEEDS[0]}"]["score_rows"])
    out["brownian_sd_f"] = sd_ref
    out["brownian_mean_f"] = float(b1.mean())
    ycap = out["yardstick_brownian_124_vs_123"]["spearman"]
    for G, E in IID_CONFIGS:
        fs = [np.array(res[k]["f"]) for s in IID_SEEDS if (k := f"iid_G{G}_E{E}_s{s}") in res]
        if not fs:
            continue
        ag = [agreement(f, b1) for f in fs]
        stack = np.stack(fs)
        row = {
            "n_seeds": len(fs),
            "spearman_vs_brownian123": [a["spearman"] for a in ag],
            "pearson_vs_brownian123": [a["pearson"] for a in ag],
            "spearman_mean": float(np.mean([a["spearman"] for a in ag])),
            "sd_f_over_brownian_sd_f": float(np.mean([f.std(ddof=1) for f in fs]) / sd_ref),   # -> lambda_s rescale
            "mean_f": float(stack.mean()),
            "seed_to_seed_sd_over_sd_p_f": float(stack.std(axis=0, ddof=1).mean() / sd_ref) if len(fs) > 1 else None,
            "score_rows_per_reward_build": int(res[f"iid_G{G}_E{E}_s{IID_SEEDS[0]}"]["score_rows"]),
        }
        row["as_good_as_brownian_yardstick"] = bool(row["spearman_mean"] >= ycap - 0.05)
        out[f"G{G}_E{E}"] = row
    return out


def print_table(summary: dict) -> None:
    y = summary["yardstick_brownian_124_vs_123"]
    print("\n" + "=" * 100)
    print(f"YARDSTICK  Brownian(124) vs Brownian(123):  Spearman {y['spearman']:.3f}  Pearson {y['pearson']:.3f}  "
          f"mean|df|/sd_p(f) {y['mean_abs_diff_over_sd']:.3f}")
    print(f"           Brownian sd_p(f) = {summary['brownian_sd_f']:.5g}   mean f = {summary['brownian_mean_f']:.5f}")
    print("-" * 100)
    print(f"{'config':10s} {'rows/build':>10s} {'Spearman (per seed)':>26s} {'mean':>6s} {'sd_f ratio':>11s} "
          f"{'seed-seed sd/sd_p':>18s}  verdict")
    for k, v in summary.items():
        if not k.startswith("G"):
            continue
        sp = " ".join(f"{s:5.2f}" for s in v["spearman_vs_brownian123"])
        ss = "-" if v["seed_to_seed_sd_over_sd_p_f"] is None else f"{v['seed_to_seed_sd_over_sd_p_f']:.3f}"
        print(f"{k:10s} {v['score_rows_per_reward_build']:>10d} {sp:>26s} {v['spearman_mean']:>6.2f} "
              f"{v['sd_f_over_brownian_sd_f']:>11.3f} {ss:>18s}  "
              f"{'>= yardstick - 0.05' if v['as_good_as_brownian_yardstick'] else 'BELOW yardstick'}")
    print("=" * 100)
    print("sd_f ratio = sd(f_IID)/sd(f_Brownian): lambda_s = 1/std_p(f) must be re-measured by this factor for IID runs.")
    print(f"Brownian reward build = {summary['brownian_score_rows_per_reward_build']} score rows; the IID rows/build above are"
          f" the saving.\nSpearman over {summary['n_heldout']} latents has SE ~0.05-0.18; treat differences < 0.1 as ties"
          f" (informative, not a test).\n")


# =====================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="CPU stand-in for FLUX; writes to a temp results file")
    args = ap.parse_args()
    global RESULTS
    if args.dry_run:
        RESULTS = os.path.join(HERE, "iid_vs_brownian_results.dryrun.json")
        if os.path.exists(RESULTS):
            os.remove(RESULTS)

    S = dry_setup() if args.dry_run else flux_setup()
    global R_REFS, N_HELDOUT
    if args.dry_run:
        R_REFS, N_HELDOUT = 16, 12
    res = load_results({**S.stamp, "R": R_REFS, "n_heldout": N_HELDOUT, "ref_seed": REF_SEED,
                        "heldout_seed": HELDOUT_SEED, "iid_configs": [list(c) for c in IID_CONFIGS],
                        "iid_seeds": list(IID_SEEDS), "brownian_seeds": list(BROWNIAN_SEEDS)})

    t0 = time.time()
    refs = S.p_gen.sample(R_REFS, seed=REF_SEED)
    X = S.p_gen.sample(N_HELDOUT, seed=HELDOUT_SEED)
    assert refs.dtype == torch.float32 and X.dtype == torch.float32, "the metric runs in fp32"
    note(f"points ready: refs {tuple(refs.shape)} heldout {tuple(X.shape)}  ({time.time() - t0:.0f}s)")

    run_g0(S, refs, res)
    if res["g0"]["verdict"] != "PASS":
        print("G0 FAILED: per-row gamma is not equivalent to per-gamma calls on this model. Continuing with G1 anyway "
              "(it uses batched_gamma=False, so it does not depend on G0).")

    for seed in BROWNIAN_SEEDS:
        run_brownian(S, refs, X, seed, res)
    for G, E in IID_CONFIGS:
        for seed in IID_SEEDS:
            run_iid(S, refs, X, G, E, seed, res)

    res["summary"] = summarize(res)
    save_results(res)
    print_table(res["summary"])
    note("done")


if __name__ == "__main__":
    main()
