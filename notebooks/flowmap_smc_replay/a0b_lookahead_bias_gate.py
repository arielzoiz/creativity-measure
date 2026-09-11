"""Gate for Fix 1 -- does the renoise lookahead RANK particles differently from the true value function?

After Step A0 closed the CRN line, one argument for Fix 1 survives: the renoise lookahead estimates
the value of a hand-made proposal (renoise by ``eta``, then a deterministic ``map``), not
``V_t(x) = log E_{z~p(x_1|x_t)}[exp(lam f(z))]``. This asks whether that distinction has any
consequence for the only thing an intermediate potential does -- **order the particles for resampling**.

THE DECISIVE NUMBER is ``spearman(V_renoise_inf, V_true)``, both at ``K -> large`` so no Monte-Carlo
noise is involved: it isolates the PROPOSAL BIAS from the estimator noise A0 already measured.

  * ~1.0  -- the proposal ranks particles exactly as the true posterior does. Fix 1 is cosmetics.
  * << 1  -- the twist is being computed on the wrong ordering. Fix 1 has a real target.

``V_true`` is the replay itself at large ``K``: Fix 1's claim is that running the base process forward
IS an exact draw from ``p(x_1|x_t)`` for the ``p`` we defined, so the large-K replay is by construction
the quantity the renoise lookahead is trying to approximate. Both are measured on the SAME particles,
taken from an untilted base run (no resampling, so lineages are the identity and ``f_final`` is
unambiguous per particle).

READ THIS RESULT ASYMMETRICALLY. ``d = 2`` understates the gap: in ``d = 65536`` a renoised Gaussian
blob and the true posterior diverge far more (concentration of measure), and the deterministic ``map``
contracts a much larger volume. So a difference here is a LOWER BOUND and promotes Fix 1; a null here
is weak and does not fully clear it. This is a screen, not a verdict.

Usage::

    conda activate creativity-measure
    python notebooks/flowmap_smc_replay/a0b_lookahead_bias_gate.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from torch import Tensor

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests"))

from test_flowmap_smc import (  # type: ignore[import-not-found]  # noqa: E402
    GaussianFlowMap, _reward,
)

from creativity_measure._types import FlowMap, Schedule  # noqa: E402
from creativity_measure.flowmap_smc import (  # noqa: E402
    LinearSchedule, _in_window, _renoise, _t_prime, _uniform_ts, ddpm_step, flow_map_step,
)
from creativity_measure.tilt import Reward  # noqa: E402

DTYPE = torch.float64
STOCH_WINDOW = (0.1, 1.0)


def _spearman(a: Tensor, b: Tensor) -> float:
    """Rank correlation -- the right metric here: resampling consumes an ORDER, not a scale."""
    ra = a.argsort().argsort().to(DTYPE)
    rb = b.argsort().argsort().to(DTYPE)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def _pearson(a: Tensor, b: Tensor) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _soft(v: Tensor, k: int) -> Tensor:
    """``log mean_k exp(v)`` over the K axis of an ``(M*K,)`` flat tensor."""
    return torch.logsumexp(v.view(-1, k), dim=1) - math.log(k)


def lookahead_renoise(
    x: Tensor, t: float, k: int, *, flow_map: FlowMap, schedule: Schedule, reward: Reward,
    lam: float, eta: float, gen: torch.Generator,
) -> Tensor:
    """Today's lookahead: renoise to ``t' < t`` with K draws, then one deterministic ``map(., t', 1)``."""
    m, d = x.shape
    t_p = _t_prime(t, eta, schedule)
    eps = torch.randn((m * k, d), generator=gen, dtype=x.dtype)
    x_rep = x.unsqueeze(1).expand(m, k, d).reshape(m * k, d)
    z = flow_map.map(_renoise(x_rep, t, t_p, schedule, eps), t_p, 1.0)
    return _soft(lam * reward(z), k)


def lookahead_replay(
    x: Tensor, n: int, ts: list[float], k: int, *, flow_map: FlowMap, schedule: Schedule,
    reward: Reward, lam: float, gen: torch.Generator,
) -> Tensor:
    """Fix 1's lookahead: finish the base process K times from ``ts[n]``, with its own noise.

    Same transitions the run itself uses (`ddpm_step` inside ``stoch_window``, `flow_map_step`
    outside), so this is a draw from ``p(x_1 | x_t)`` for the ``p`` actually defined -- which is the
    whole of Fix 1's argument.
    """
    m, d = x.shape
    z = x.unsqueeze(1).expand(m, k, d).reshape(m * k, d)
    for j in range(n, len(ts) - 1):
        step = ddpm_step if _in_window(ts[j], STOCH_WINDOW) else flow_map_step
        z = step(z, ts[j], ts[j + 1], flow_map=flow_map, schedule=schedule, generator=gen)
    return _soft(lam * reward(z), k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lams", type=float, nargs="*", default=[4.0, 8.0, 16.0])
    ap.add_argument("--n-particles", type=int, default=64)
    ap.add_argument("--k-big", type=int, default=2048, help="K for the noise-free reference")
    ap.add_argument("--k-run", type=int, default=4, help="K a real run would use")
    ap.add_argument("--eta", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    schedule = LinearSchedule()
    flow_map = GaussianFlowMap(schedule)
    reward = _reward()
    ts = _uniform_ts(16)
    m, d = args.n_particles, 2

    # One untilted base cloud, shared by every estimator and every lambda. No resampling anywhere, so
    # each particle owns its endpoint and `f_final` needs no lineage bookkeeping.
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn((m, d), generator=gen, dtype=DTYPE)
    cloud: list[tuple[int, float, Tensor]] = []
    for n in range(len(ts) - 1):
        step = ddpm_step if _in_window(ts[n], STOCH_WINDOW) else flow_map_step
        x = step(x, ts[n], ts[n + 1], flow_map=flow_map, schedule=schedule, generator=gen)
        cloud.append((n + 1, ts[n + 1], x.clone()))
    f_final = reward(cloud[-1][2]).to(DTYPE)

    print(f"toy: p = N(0, 1.3^2 I), d={d}, M={m}, N={len(ts) - 1}, stoch_window={STOCH_WINDOW}, "
          f"K_big={args.k_big}, K_run={args.k_run}, eta={args.eta}")
    print("V_true := replay at K_big.  V_renoise_inf := renoise at K_big (proposal bias, NO MC noise).")

    for lam in args.lams:
        print(f"\n{'=' * 104}\nlam = {lam:g}")
        print(f"{'t':>7} {'a=lam*sd_k':>10} | {'BIAS ONLY: rank(V_ren_inf,V_true)':>33} "
              f"{'pearson':>8} | {'rank(V_ren_K4,V_true)':>21} | "
              f"{'corr(V_true,f_fin)':>18} {'corr(V_ren,f_fin)':>17}")
        rows: list[tuple[float, float, float, float, float]] = []
        for n, t, xt in cloud[:-1]:                      # the terminal step has no lookahead
            g1 = torch.Generator().manual_seed(1000 + n)
            g2 = torch.Generator().manual_seed(2000 + n)
            g3 = torch.Generator().manual_seed(3000 + n)
            v_true = lookahead_replay(xt, n, ts, args.k_big, flow_map=flow_map, schedule=schedule,
                                      reward=reward, lam=lam, gen=g1)
            v_ren_inf = lookahead_renoise(xt, t, args.k_big, flow_map=flow_map, schedule=schedule,
                                          reward=reward, lam=lam, eta=args.eta, gen=g2)
            v_ren_k = lookahead_renoise(xt, t, args.k_run, flow_map=flow_map, schedule=schedule,
                                        reward=reward, lam=lam, eta=args.eta, gen=g3)
            a = lam * float(reward(
                _renoise(xt.unsqueeze(1).expand(-1, args.k_run, -1).reshape(-1, d), t,
                         _t_prime(t, args.eta, schedule), schedule,
                         torch.randn((m * args.k_run, d), generator=torch.Generator().manual_seed(n),
                                     dtype=DTYPE))
            ).view(m, args.k_run).std(dim=1).mean())

            sp_bias = _spearman(v_ren_inf, v_true)
            pe_bias = _pearson(v_ren_inf, v_true)
            sp_k = _spearman(v_ren_k, v_true)
            c_true = _pearson(v_true, f_final)
            c_ren = _pearson(v_ren_k, f_final)
            rows.append((sp_bias, pe_bias, sp_k, c_true, c_ren))
            print(f"{t:>7.4f} {a:>10.3f} | {sp_bias:>33.3f} {pe_bias:>8.3f} | {sp_k:>21.3f} | "
                  f"{c_true:>18.3f} {c_ren:>17.3f}")

        def _mean(i: int, lo: int = 0, hi: int = 10 ** 9) -> float:
            vals = [r[i] for j, r in enumerate(rows) if r[i] == r[i] and lo <= j < hi]
            return sum(vals) / len(vals) if vals else float("nan")

        # Split at t = 0.25. The mean over all 15 steps is dominated by the late half, where every
        # estimator agrees trivially (the posterior has collapsed and V -> lam*f), and that hides the
        # only region where the two proposals differ -- which is also where the FLUX runs collapse.
        n_early = sum(1 for n, t, _ in cloud[:-1] if t <= 0.25)
        print(f"{'MEAN all':>8} {'':>9} | {_mean(0):>33.3f} {_mean(1):>8.3f} | {_mean(2):>21.3f} | "
              f"{_mean(3):>18.3f} {_mean(4):>17.3f}")
        print(f"{'t<=0.25':>8} {'':>9} | {_mean(0, 0, n_early):>33.3f} "
              f"{_mean(1, 0, n_early):>8.3f} | {_mean(2, 0, n_early):>21.3f} | "
              f"{_mean(3, 0, n_early):>18.3f} {_mean(4, 0, n_early):>17.3f}")
        print(f"{'t>0.25':>8} {'':>9} | {_mean(0, n_early):>33.3f} {_mean(1, n_early):>8.3f} | "
              f"{_mean(2, n_early):>21.3f} | {_mean(3, n_early):>18.3f} {_mean(4, n_early):>17.3f}")
        early = _mean(0, 0, n_early)
        print(f"  -> proposal bias is {'REAL and early' if early < 0.9 else 'negligible'} "
              f"(rank {early:.3f} at t<=0.25, {_mean(0, n_early):.3f} after). "
              f"Endpoint relevance t<=0.25: V_true {_mean(3, 0, n_early):.3f} vs "
              f"V_renoise {_mean(4, 0, n_early):.3f} -- "
              f"{'no benefit from being correct' if _mean(4, 0, n_early) >= _mean(3, 0, n_early) else 'truth predicts better'}")


if __name__ == "__main__":
    main()
