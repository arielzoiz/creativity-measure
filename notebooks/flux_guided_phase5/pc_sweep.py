"""Phase 5: does Langevin correction extend Phase 3's creative window?

Phase 3 (`notebooks/flux_guided_phase3/fine_lambda_sweep.py`, jobs 958150/958630) found a
creative-but-recognizable window at lam in [0.4, 3.54] and a hard ceiling past lam ~ 3.93, where the
guided ODE locks onto a fixed off-manifold attractor and f runs away (143.97 at lam=3.93 -> 271.24 at
lam=5.5, all visually deep-fried). Phase 5 interleaves `corrector_steps` ULA steps at each arrival node, whose drift contains the model's own marginal score, so the particle re-equilibrates onto p_t after
each nudge (`creativity_measure/samplers/flow_guided_pc.py`).

THREE ARMS, one job each, same reward / same refs / same z0 / same GPU model:

    pc_guided     predictor_guided=True,  corrector_steps=C   -- strict superset of Phase 3
    pc_unguided   predictor_guided=False, corrector_steps=C   -- all tilt from the corrector
    flow_guided   corrector_steps=0                           -- Phase 3

The third arm is RE-RUN here rather than read out of Phase 3's stored JSON on purpose: CLAUDE.md records
that a GPU model change alone shifts f by 16% of std_p(f) (job 697271), the same order as the effect being
measured. That it IS Phase 3 is established two ways, deliberately split by what each can actually prove:
backend-independently in CI, by comparing traced call sequences (test_flow_guided_pc.py's
`..._requests_an_identical_CALL_SEQUENCE`); and on this hardware at startup, bitwise at lam=0 where no
backward is taken. At lam != 0 the deviation is RECORDED against the backend's own self-reproducibility
floor rather than asserted -- FLUX's backward is not bit-reproducible on GPU, which cost jobs
965868/965869 before the check was corrected. See `preflight` item (5).

WHAT SUCCESS LOOKS LIKE -- rewritten after wave 1, because the original text here predicted the opposite
of what happened and named the wrong mechanism.

The corrector does NOT pull back toward p_t. It targets q_t ~ p_t exp(lam r), which at any meaningful lam
is itself off-manifold, so the corrector ACCELERATES the tilt: at matched lam the PC arms report HIGHER f
(+65% at lam=1.57, +21% at lam=2.36), not lower.

**LOOK AT THE DECODED PNGs FIRST.** They are written per point the moment it completes, and they are the
only thing that answers the real question. Wave 1 spent hours on scalar comparisons that pointed the wrong
way; five image reads reversed two conclusions. Specifically:
  - `x_norm_final` is ANTI-correlated with quality here: the destroyed n_steps=19 image measures 2.88
    against the intact PC image's 3.90. Do NOT read low ||x|| as "on-manifold". It is recorded because it
    is worth having, not because it is a quality proxy.
  - `hf_frac` (spectral power above 0.25 Nyquist) is the LEAST bad scalar, but it is not a predictor
    either: it ranks arms sensibly within one seed at one lam, yet has no absolute threshold across
    seeds -- seed 1234 scores 0.0089 on a recognizable dog at lam=1.571 while seed 3141 scores 0.0093 on
    a destroyed mosaic. Use it only as a within-comparison ranking, never as "below X means intact".
  - `n_steps` must be MATCHED across arms. Raising it does not merely cost more compute, it destroys
    guidance (n_steps=19 is abstract blocks at lam=1.571 where n_steps=10 is a recognizable dog), so it
    cannot be used to build a compute-matched control.

Wave 1's answer, at matched lam and matched n_steps=10: lam=1.571 both recognizable (f 14.2 vs 23.5);
lam=2.357 Phase 3 degraded to a pictograph while PC is still clearly a dog (f 47.2 vs 57.2); lam=3.143
both broken. About one lam step of extra structural integrity plus 20-65% more novelty, at 1.9x compute.
Same rule as everywhere else in this repo: rising novelty is never the stopping signal.

Setup is IMPORTED from Phase 3's fine_lambda_sweep.py rather than copied: lambda_s, the reference latents
and the whole reward config must be bit-identical to that run for the cross-phase image comparison to
mean anything, and a copy would drift. Everything imported is construction-only -- that module's own
main() is __main__-guarded and never runs here.

CONVENTION NOTE: t is diffusers-native (t=1 noise, t=0 data) throughout, same as flow_guided.py.
Read E_q[f] at the terminal t=0 only.

    python pc_sweep.py --arm pc_guided          # on a GPU node (see pc_sweep.slurm)
    python pc_sweep.py --arm pc_guided --dry-run  # CPU, tiny FluxTransformer2DModel, no decode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")
sys.path.insert(0, PHASE3)

from fine_lambda_sweep import (   # noqa: E402  # pyright: ignore[reportMissingImports]  (sys.path above)
    PROMPT as PROMPT_DEFAULT, N_PARTICLES, N_STEPS_ODE, SHIFT, SWEEP_SEED, Setup, dry_setup,
    flux_setup, note,
)

from creativity_measure.samplers.flow_guided import flow_guided_sample                     # noqa: E402
from creativity_measure.samplers.flow_guided_pc import (                                   # noqa: E402
    SNR_SONG_2021, flow_guided_pc_sample, velocity_to_score,
)

# Lambda lattice: Phase 3's zoom grid is 15 points uniform in [0, 5.5], i.e. k * LAM_STEP for
# k = 0..14. Addressing lambdas by lattice index k keeps every point ON that lattice -- so each lambda
# <= 5.5 lands exactly on an existing fine_decoded_max5.5/*.png and the images pair one-to-one -- while
# letting k > 14 extend PAST Phase 3's range. That extension is the point: the max5.5 grid was chosen to
# bracket PHASE 3's collapse, so if the corrector pushes the ceiling out, 5.5 is too short to find it.
LAM_STEP_PHASE3 = 5.5 / 14
LAM_STEP = LAM_STEP_PHASE3     # overridable per run via --lam-step; see main()

# The prompt study (2026-10-07) uses --lam-step 0.2 with k=1..5, i.e. lambda in {0.2,...,1.0}. That
# lattice does NOT pair with Phase 3's images and is not meant to: past lam ~ 1 both arms are already
# too noisy to read, and the whole measured benefit lives below it. lam=0 is the unguided sample, so it
# is the "clear" image for that (prompt, seed) and is method-independent.
LAM_STEP_PROMPT_STUDY = 0.2

# Wave 1: uniform every-other-lattice-point from 0 to 7.857. Deliberately NOT concentrated on the
# high-lam region, for two reasons found while checking a sparser draft:
#   (a) Phase 3's creative window [0.40, 3.54] holds 8 lattice points (k=2..9); the corrector may well
#       change behaviour INSIDE it (it pulls back toward p_t, so it could make images less creative, not
#       more), and a grid that samples the window twice cannot see that.
#   (b) under eta_reference="total" the corrector's measured displacement falls 12x from lam=0.79 to
#       lam=5.5 (dry run: 0.288 -> 0.024; faster than the 1/(1+lam) coupling alone predicts, because
#       ||s_theta|| also grows as the latents drift off-manifold). So at high lam the corrector is nearly
#       switched off, and "PC ~ Phase 3 up there" would be an artifact of the step-size rule rather than a
#       statement about the mechanism. Hence the companion eta_reference="score" arm, where eta is
#       lam-independent and lam only ROTATES the drift.
LAM_K_DEFAULT = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)

CORRECTOR_STEPS = 1            # ULA steps per ODE step. NOT this repo's m = lam/lam_s, and not
                               # Algorithm 3's particle count M -- both already exist in CLAUDE.md.
                               # 1 is Song et al.'s reference PC setting at snr=0.16, and it makes
                               # pc_unguided cost 524 s/lambda against Phase 3's measured 572 s/lambda --
                               # i.e. the EXISTING max5.5 run is already a compute-matched control for
                               # that arm, for free.
SNR = SNR_SONG_2021
ETA_REFERENCE = "total"
EXACT_JACOBIAN = True          # Phase 3's production setting (fine_lambda_sweep.py), for comparability

ARMS = ("pc_guided", "pc_unguided", "flow_guided")

# Preflight reference values. lambda_s is PER PROMPT, not a constant: the reference latents are G(z)
# under the prompt-conditioned velocity, so a new prompt is a new reward with its own std_p(f). The
# registry below is seeded with Phase 3's measured value for its own prompt; any prompt not in it is
# MEASURED and recorded on first use, and asserted against that recorded value on every later job. The
# check is a config-drift guard ("did the refset/reward change under me"), which is exactly as useful
# per prompt as it was against a hard-coded 81.016 -- it just cannot be known before the first run.
LAM_S_SEED_VALUES = {"A dog": 81.016}          # every Phase 3 seed's stamp, on an NVIDIA L40S
LAM_S_REGISTRY = os.path.join(HERE, "lam_s_by_prompt.json")
LAM_S_TOL = 0.05
EXPECTED_GPU = "L40S"

LAM_GRID: list[float] = []     # set in main() from --lam-k
N_STEPS: int = 0               # set in main() from --n-steps
RESULTS = ""                   # set in main(), keyed on the FULL config
DECODED_DIR = ""


# =====================================================================================================
# Image-space off-manifold proxy
# =====================================================================================================

def hf_power_fraction(img: Tensor, cutoff: float = 0.25) -> float:
    """Fraction of 2D spectral power at radial frequency above ``cutoff`` x Nyquist.

    The quantitative stand-in for "is it deep-fried": CLAUDE.md's gamma-window finding already used
    exactly this band (samples gained 1.5-1.7x more power above 0.25 Nyquist when the lower gamma cut was
    extended, i.e. high-frequency graininess). DC is zeroed so the measure is contrast-invariant.

    Args:
        img: ``(C, H, W)`` or ``(H, W)`` in [0, 1]. Channels are averaged to luminance first.
    """
    g = img.mean(dim=0) if img.ndim == 3 else img
    p = torch.fft.fftshift(torch.fft.fft2(g.double())).abs() ** 2
    h, w = p.shape
    fy = torch.fft.fftshift(torch.fft.fftfreq(h)).abs() * 2.0        # in units of Nyquist
    fx = torch.fft.fftshift(torch.fft.fftfreq(w)).abs() * 2.0
    r = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
    p[h // 2, w // 2] = 0.0                                          # drop DC
    total = float(p.sum())
    return float(p[r > cutoff].sum()) / total if total > 0 else float("nan")


# =====================================================================================================
# Results file (atomic, resumable per lambda) -- same convention as Phase 3's fine_lambda_sweep.py
# =====================================================================================================

# Stamp fields that must match for two runs to count as "the same sweep" for resume/merge purposes.
# Deliberately EXCLUDES:
#   - lam_k / lam_grid: which lattice points THIS invocation asked for, not what any point means. A job
#     requesting a different subset of the same lattice (e.g. filling in the points an earlier job
#     skipped) must merge into the existing file, not discard it -- this is the exact failure that cost a
#     full C=1 re-run across 5 seeds in the corrector_steps sweep (2026-10-03, see NEXT_SESSION.md).
#   - preflight: a fresh on-hardware measurement every run, and PARTLY NONDETERMINISTIC by construction
#     (nondet_floor_lam1 / reduction_cross_lam1 -- CLAUDE.md: FLUX's backward is not bit-reproducible on
#     GPU). Requiring it to match means resume would almost never fire, even for a literal rerun of
#     identical args.
_VOLATILE_STAMP_KEYS = ("lam_k", "lam_grid", "preflight")


def _stamp_identity(stamp: dict) -> dict:
    return {k: v for k, v in stamp.items() if k not in _VOLATILE_STAMP_KEYS}


def load_results(stamp: dict) -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            old = json.load(fh)
        old_stamp = old.get("stamp", {})
        if _stamp_identity(old_stamp) == _stamp_identity(stamp):
            # Merge: the file's lam_k/lam_grid become the union of what it already had and what this
            # run asked for. Per-point keys are addressed by lattice index (see run_one), so merged
            # entries from different --lam-k subsets never collide.
            merged_lam_k = sorted(set(old_stamp.get("lam_k", [])) | set(stamp.get("lam_k", [])))
            old["stamp"] = {**stamp, "lam_k": merged_lam_k,
                             "lam_grid": [k * LAM_STEP for k in merged_lam_k]}
            print(f"[resume] {RESULTS}: {sorted(k for k in old if k != 'stamp')}")
            return old
        os.replace(RESULTS, RESULTS + ".stale")
        print("[resume] stamp mismatch on identity fields -> old results moved to .stale, starting fresh")
    return {"stamp": stamp}


def save_results(res: dict) -> None:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2)
    os.replace(tmp, RESULTS)


# =====================================================================================================
# The sweep
# =====================================================================================================

def prompt_slug(prompt: str) -> str:
    """Filename-safe token for a prompt, so two prompts never share a results file or decode dir."""
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in prompt).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "empty"


def _lam_s_registry() -> dict[str, float]:
    reg = dict(LAM_S_SEED_VALUES)
    if os.path.exists(LAM_S_REGISTRY):
        with open(LAM_S_REGISTRY) as fh:
            reg.update(json.load(fh))
    return reg


def _record_lam_s(prompt: str, lam_s: float) -> None:
    """Atomic, and last-writer-wins on purpose.

    The two jobs of a prompt (one per seed) run concurrently and both measure lambda_s before either
    has recorded it, so both will write. That is harmless: they ran the same deterministic setup on the
    same pinned GPU model, so they write the same number to within the tolerance the value is used at.
    """
    reg = {}
    if os.path.exists(LAM_S_REGISTRY):
        with open(LAM_S_REGISTRY) as fh:
            reg = json.load(fh)
    reg[prompt] = lam_s
    tmp = f"{LAM_S_REGISTRY}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(reg, fh, indent=2, sort_keys=True)
    os.replace(tmp, LAM_S_REGISTRY)


def _arm_kwargs(arm: str) -> dict[str, Any]:
    if arm == "pc_guided":
        return dict(predictor_guided=True, corrector_steps=CORRECTOR_STEPS)
    if arm == "pc_unguided":
        return dict(predictor_guided=False, corrector_steps=CORRECTOR_STEPS)
    if arm == "flow_guided":
        return dict(predictor_guided=True, corrector_steps=0)
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def preflight(S: Setup, sweep_seed: int, *, dry_run: bool, prompt: str = PROMPT_DEFAULT) -> dict[str, Any]:
    """Tier-1 checks: everything whose failure invalidates the WHOLE job. Raises; never warns.

    Runs before any sweep point. Total cost ~4 min against a multi-hour sweep, and all but the last two
    items are free (they ride on values the setup already computed).
    """
    out: dict[str, Any] = {}

    # (1) GPU model. The arms are only comparable on ONE model -- a GPU change alone shifts f by 16% of
    # std_p(f) (job 697271), the same order as the effect being measured. Checked FIRST because it is the
    # only item that does not need the reward, i.e. the only one that can fail before the ref-bank build.
    if not dry_run:
        gpu = torch.cuda.get_device_name(0)
        out["gpu_name"] = gpu
        note(f"preflight gpu: {gpu}")
        if EXPECTED_GPU not in gpu:
            raise AssertionError(
                f"expected a {EXPECTED_GPU} (every Phase 3 run's stamp records 'NVIDIA L40S') but landed "
                f"on {gpu!r}. Results from this GPU are NOT comparable to the other arms -- fix "
                "--constraint and resubmit rather than interpreting this run."
            )

    # (2) f(x_refs) == (R-1)/R exactly, for uniform weights (CLAUDE.md). Free: the bank is already built,
    # so this costs no score rows. Tolerance 1e-3, NOT 0.10 -- 0.10 cannot separate 63/64 from the
    # failure modes that land on exactly 1.0. Tests normalization wiring only.
    with torch.no_grad():
        f_refs = float(S.reward(S.reward.x_refs).mean())
    r = S.reward.x_refs.shape[0]
    expected = (r - 1) / r
    out["f_refs"] = f_refs
    note(f"preflight f(refs): {f_refs:.6f} vs (R-1)/R = {expected:.6f}  (R={r})")
    if abs(f_refs - expected) > 1e-3:
        raise AssertionError(
            f"f(x_refs) = {f_refs:.6f} but uniform weights demand exactly (R-1)/R = {expected:.6f}. "
            "The reward's normalization is mis-wired; every f in this job would be on the wrong scale."
        )

    # (3) lambda_s reproduces THIS PROMPT's recorded value. Free (already computed by
    # _build_reward_and_lam_s). This is the check that the refset/reward config did not drift from the
    # run we are comparing against. A prompt with no recorded value is measured and recorded here -- it
    # cannot be asserted on its first job, by construction.
    out["lam_s"] = S.lam_s
    if not dry_run:
        known = _lam_s_registry().get(prompt)
        if known is None:
            note(f"preflight lambda_s: {S.lam_s:.3f} -- first run for prompt {prompt!r}, recording it "
                 "as this prompt's reference (no assert possible on a first run)")
            _record_lam_s(prompt, S.lam_s)
        else:
            rel = abs(S.lam_s - known) / known
            note(f"preflight lambda_s: {S.lam_s:.3f} vs {prompt!r}'s recorded {known:.3f} "
                 f"(rel {rel:.2%})")
            if rel > LAM_S_TOL:
                raise AssertionError(
                    f"lambda_s = {S.lam_s:.3f} differs from the recorded value for prompt {prompt!r} "
                    f"({known:.3f}) by {rel:.1%} (> {LAM_S_TOL:.0%}). The reference latents or the "
                    f"reward config diverged, so no number here is comparable to that run. If the "
                    f"divergence is intended, delete {prompt!r} from "
                    f"{os.path.basename(LAM_S_REGISTRY)}."
                )

    # (4) The score reparametrization, cross-validated against the reward's OWN EDM/Tweedie score path on
    # the real model. The formula is pure algebra and already unit-tested on CPU; what this buys is
    # NUMERICAL CONDITIONING in bfloat16 at d=65536, since eps_hat = x_t + (1-t)v is a difference of
    # same-order terms. Two independently-coded paths, related by the Jacobian of x_sigma = x_t/(1-t):
    #     velocity_to_score(x_t, v, t)  ==  gamma * score_fn(gamma * x_t/(1-t), gamma) / (1-t)
    # t is swept over the corrector's actual operating range; gamma must stay inside the denoiser's
    # [1/sigma_max^2, 1/sigma_min^2], which excludes t -> 1.
    score_fn = getattr(S.reward.distance, "score_fn", None)
    if score_fn is None:
        note("preflight score check: SKIPPED (distance exposes no score_fn)")
    else:
        t0 = time.time()
        gen = torch.Generator(device=S.device).manual_seed(4242)
        rels: dict[str, float] = {}
        for t in (0.9, 0.5, 0.1, 0.02):
            x_t = torch.randn(1, S.d, generator=gen, device=S.device, dtype=S.dtype)
            with torch.no_grad():
                v = S.velocity_fn(x_t, t)
                s_flow = velocity_to_score(x_t, v, t)
                gamma = torch.tensor(((1.0 - t) / t) ** 2, device=S.device, dtype=torch.float32)
                s_edm = gamma * score_fn(gamma * x_t.float() / (1.0 - t), gamma) / (1.0 - t)
            rel = float((s_flow - s_edm).norm() / s_edm.norm().clamp_min(1e-30))
            rels[f"t{t}"] = rel
            note(f"preflight score check t={t:.2f}: rel diff {rel:.3e}")
        out["score_xval_rel"] = rels
        out["score_xval_s"] = time.time() - t0
        worst = max(rels.values())
        if worst > 2e-2:
            raise AssertionError(
                f"velocity_to_score disagrees with the reward's own EDM score path by {worst:.3e} "
                "(> 2e-2) at production scale/dtype. The corrector's drift and the reward would be using "
                "inconsistent scores; do not interpret this run."
            )

    # (5) The reduction, asserted ONLY where the hardware can actually deliver it.
    #
    # The three-arm comparison rests on the flow_guided arm BEING Phase 3 rather than merely
    # Phase-3-like. The first version of this check asserted that bitwise at lam=1 with
    # exact_jacobian=True, and jobs 965868/965869 both died on it (max abs diff 1.17e+01) even though
    # tests/test_flow_guided_pc.py asserts exactly that, bitwise, on CPU in both Jacobian modes.
    #
    # The check was wrong, not the code. Bitwise-output equality conflates two claims:
    #   (i)  the two samplers ask the model the SAME question -- a property of this repo, now asserted
    #        permanently and backend-independently by
    #        test_corrector_steps_zero_requests_an_identical_CALL_SEQUENCE, which compares traced
    #        (t, shape, input-hash, requires_grad) sequences and finds them identical at every lam in
    #        both Jacobian modes;
    #   (ii) the model returns the SAME answer to the same question twice -- a property of the backend,
    #        which FLUX's backward on GPU does not have (flash-attention atomics plus the
    #        gradient-checkpoint recompute `_freeze` enables).
    # Only (i) is ours to assert. Note what Phase 3 verified on hardware (job 957386): "lam=0 bitwise
    # parity holds on real hardware too" -- at lam=0, where no backward is taken. That case IS bitwise
    # and is asserted below; it is also cheap (2 unguided steps).
    #
    # For lam != 0 the cross-sampler deviation is RECORDED next to the model's own self-deviation floor
    # (same function, same seed, twice) rather than asserted: a deviation at the floor is the backend,
    # and one far above it would be news. Making this a measurement instead of a gate is what stops a
    # backend property from blocking 20+ GPU-hours of otherwise-valid sampling.
    kw0: dict[str, Any] = dict(
        velocity_fn=S.velocity_fn, n_steps=2, shift=SHIFT, t_start=1.0, t_end=0.0,
        exact_jacobian=EXACT_JACOBIAN, seed=sweep_seed,
    )
    t0 = time.time()
    ref0 = flow_guided_sample(S.reward, 0.0, N_PARTICLES, **kw0)
    pc0 = flow_guided_pc_sample(S.reward, 0.0, N_PARTICLES, corrector_steps=0, **kw0)
    ok0 = torch.equal(pc0.X, ref0.X)
    out["reduction_bitwise_lam0"] = ok0
    note(f"preflight reduction (lam=0, no backward -> must be bitwise): {ok0} ({time.time() - t0:.1f}s)")
    if not ok0:
        raise AssertionError(
            "flow_guided_pc_sample(corrector_steps=0) is NOT bitwise equal to flow_guided_sample at "
            f"lam=0 (max abs diff {float((pc0.X - ref0.X).abs().max()):.3e}). No backward is taken at "
            "lam=0, so this path IS deterministic and a mismatch here is a REAL defect or a config "
            "drift -- not backend nondeterminism. Do not interpret any result from this job."
        )

    t0 = time.time()
    a = flow_guided_sample(S.reward, 1.0, N_PARTICLES, **kw0)
    b = flow_guided_sample(S.reward, 1.0, N_PARTICLES, **kw0)
    c = flow_guided_pc_sample(S.reward, 1.0, N_PARTICLES, corrector_steps=0, **kw0)
    floor = float((a.X.float() - b.X.float()).abs().max())      # the backend's own reproducibility
    cross = float((a.X.float() - c.X.float()).abs().max())
    out["nondet_floor_lam1"] = floor
    out["reduction_cross_lam1"] = cross
    note(f"preflight reduction (lam=1, exact_jacobian={EXACT_JACOBIAN}): self-deviation floor "
         f"{floor:.4e}, cross-sampler {cross:.4e}, "
         + (f"ratio {cross / floor:.2f}" if floor > 0 else
            ("BOTH EXACT" if cross == 0 else "floor=0 but cross>0 -- see assert below"))
         + f"  ({time.time() - t0:.1f}s)")
    if floor == 0.0 and cross > 0.0:
        raise AssertionError(
            f"the backend IS bit-reproducible here (self-deviation 0) yet the two samplers differ by "
            f"{cross:.3e}. That cannot be nondeterminism -- it is a real difference between the paths."
        )
    return out


def check_point(result: Any, arm: str, lam: float, n_corr_steps: int) -> None:
    """Tier-2 asserts: free, per-point, and they RAISE rather than warn.

    The sweep is resumable per lambda, so a hard failure costs one point and a resubmit; a silently wrong
    point pollutes a cross-arm conclusion that someone will act on. Every quantity here is already
    recorded -- nothing extra is computed.
    """
    if not torch.isfinite(result.X).all():
        raise AssertionError(f"lam={lam}: terminal latents contain non-finite values")

    ran = [s for s in result.corrector_history if s.ran]
    n_corr_nodes = len(ran)

    # Effort accounting: the cheapest possible check that the sampler did what its kwargs said. The
    # corrector is skipped at the schedule's final t=0 node, so it is n_corr_nodes, NOT n_steps.
    exp_vel = N_STEPS + n_corr_nodes * n_corr_steps
    exp_grad = (N_STEPS if arm != "pc_unguided" and lam != 0 else 0) + (
        n_corr_nodes * n_corr_steps if lam != 0 else 0)
    if result.n_velocity_evals != exp_vel:
        raise AssertionError(
            f"lam={lam}: {result.n_velocity_evals} velocity evals, expected {exp_vel} "
            f"(= {N_STEPS} predictor + {n_corr_nodes} nodes x {n_corr_steps})")
    if result.n_reward_grads != exp_grad:
        raise AssertionError(
            f"lam={lam}: {result.n_reward_grads} reward grads, expected {exp_grad}")

    if lam == 0.0 and result.n_reward_grads != 0:
        raise AssertionError(f"lam=0 must cost zero reward gradients, got {result.n_reward_grads}")

    for s in ran:
        # ||g_tilde|| == ||s_theta|| by construction, so ||g_total|| <= (1+lam)||s_theta|| exactly.
        for dn, sn in zip(s.drift_norm, s.score_norm):
            if dn > (1.0 + abs(lam)) * sn * 1.001 + 1e-6:
                raise AssertionError(
                    f"lam={lam}, t={s.t}: ||g_total||={dn:.6g} exceeds (1+lam)||s||={(1+abs(lam))*sn:.6g}")
        # sqrt(2 eta)||z|| / (eta ||g_total||) == 1/snr identically under eta_reference="total".
        if ETA_REFERENCE == "total":
            for nf in s.noise_frac:
                if abs(nf - 1.0 / SNR) > 1e-3 * (1.0 / SNR):
                    raise AssertionError(
                        f"lam={lam}, t={s.t}: noise/drift ratio {nf:.6f} != 1/snr = {1.0/SNR:.6f}; "
                        "the step size is mis-wired")


def _z0_for(S: Setup, z0_seed: int) -> Tensor:
    """The same z0 ``flow_guided_pc_sample(seed=z0_seed)`` would draw for itself.

    Passing it explicitly is what lets a run hold z0 FIXED while ``--sweep-seed`` varies only the
    corrector noise -- the decomposition that separates PC's extra (Langevin) variance from the z0
    variance Phase 3 also has. The draw must match the sampler's own line exactly (same shape, device,
    dtype and generator seeding) or the two are not comparable.
    """
    gen = torch.Generator(device=S.device).manual_seed(z0_seed)
    return torch.randn(N_PARTICLES, S.d, generator=gen, device=S.device, dtype=S.dtype)


def run_one(S: Setup, arm: str, idx: int, lam: float, res: dict, sweep_seed: int,
            z0_seed: int | None) -> None:
    # Keyed by the LATTICE index (idx == k from --lam-k), not this job's own positional enumerate()
    # index -- so the same lambda always lands on the same key no matter which subset of the lattice a
    # given job requested, and merged entries from different --lam-k invocations never collide.
    key = f"idx{idx:02d}_lam{lam:.4f}"
    if key in res:
        note(f"{key} already done: f={res[key]['f_mean']:.5f}, t={res[key]['t_sample_s']:.1f}s")
        return

    t0 = time.time()
    result = flow_guided_pc_sample(
        S.reward, lam, N_PARTICLES, velocity_fn=S.velocity_fn, n_steps=N_STEPS, shift=SHIFT,
        t_start=1.0, t_end=0.0, exact_jacobian=EXACT_JACOBIAN, snr=SNR, eta_reference=ETA_REFERENCE,
        seed=sweep_seed, z0=None if z0_seed is None else _z0_for(S, z0_seed), **_arm_kwargs(arm),
    )
    t_sample = time.time() - t0
    check_point(result, arm, lam, _arm_kwargs(arm)["corrector_steps"])
    with torch.no_grad():
        f = S.reward(result.X)

    ran = [s for s in result.corrector_history if s.ran]
    flat = lambda attr: [v for s in ran for v in getattr(s, attr)]           # noqa: E731
    mean = lambda vals: (sum(vals) / len(vals)) if vals else float("nan")    # noqa: E731

    png_path, hf_frac = None, float("nan")
    if S.decode is not None:
        os.makedirs(DECODED_DIR, exist_ok=True)
        img = S.decode(result.X)[0]
        hf_frac = hf_power_fraction(img)
        from PIL import Image
        arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        png_path = os.path.join(DECODED_DIR, f"{key}.png")
        Image.fromarray(arr).save(png_path)

    res[key] = {
        "arm": arm, "idx": idx, "lam": lam, "m": lam / S.lam_s if S.lam_s else 0.0,
        "f_mean": float(f.mean()), "f_std": float(f.std()),
        # predictor diagnostics -- same fields Phase 3 logs, so the arms line up column for column
        "applied_norm_mean": mean(result.applied_norm_history),
        "v_norm_mean": mean(result.v_norm_history),
        # corrector diagnostics
        "corrector_nodes": len(ran),
        "corrector_eta_mean": mean(flat("eta")),
        "corrector_rel_disp_mean": mean(flat("rel_displacement")),
        "corrector_score_norm_mean": mean(flat("score_norm")),
        "corrector_drift_norm_mean": mean(flat("drift_norm")),
        "corrector_noise_frac_mean": mean(flat("noise_frac")),   # must be 1/snr = 6.25 under "total"
        # off-manifold proxies
        "x_norm_final": result.x_norm_history[-1] if result.x_norm_history else float("nan"),
        "hf_frac": hf_frac,
        # accounting
        "n_velocity_evals": result.n_velocity_evals,
        "n_reward_grads": result.n_reward_grads,
        "oom_fallback_any": (any(result.oom_fallback_history)
                              or any(f for s in result.corrector_history for f in s.oom_fallback)),
        "t_sample_s": t_sample,
        "png": png_path,
    }
    save_results(res)
    r = res[key]
    note(f"{key}: m={r['m']:.3f} f={r['f_mean']:.4f} |x|/sqrt(d)={r['x_norm_final']:.4f} "
         f"hf={r['hf_frac']:.4f} eta={r['corrector_eta_mean']:.3g} "
         f"disp={r['corrector_rel_disp_mean']:.3g} nf={r['corrector_noise_frac_mean']:.3f} "
         f"t={t_sample:.1f}s ({t_sample / N_STEPS:.2f}s/ode-step)")


def main() -> None:
    global RESULTS, DECODED_DIR, CORRECTOR_STEPS, SNR, ETA_REFERENCE, LAM_GRID, N_STEPS, LAM_STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, required=True,
                    help=f"comma-separated arms from {ARMS}. Several arms in ONE job share the model "
                         "load, the reference bank and the preflight -- measured at 35-90 min on this "
                         "path, i.e. comparable to the sweep itself -- and are guaranteed to land on "
                         "the same GPU, which the cross-arm comparison requires anyway.")
    ap.add_argument("--prompt", type=str, default=PROMPT_DEFAULT,
                    help="text prompt (default: %(default)r, Phase 3's). A prompt is a SEPARATE REWARD: "
                         "the reference latents are G(z) under the prompt-conditioned velocity, so "
                         "S_scale, gamma_lo and lambda_s all move with it. It enters the results key, "
                         "so prompts never share a file.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", type=str, default=None,
                    help="extra suffix on top of the auto-generated config key (see below). Only needed "
                         "to separate two runs that differ in NOTHING the key already captures.")
    ap.add_argument("--n-steps", type=int, default=N_STEPS_ODE,
                    help="ODE steps (default: %(default)s, Phase 3's). Raise it on the flow_guided arm "
                         "to build a COMPUTE-MATCHED control: pc_guided at n_steps=10, corrector_steps=1 "
                         "spends 10 + 9 = 19 guided units, so flow_guided at --n-steps 19 costs the same "
                         "and isolates 'the corrector helps' from 'more compute helps'.")
    ap.add_argument("--lam-k", type=str, default=",".join(str(k) for k in LAM_K_DEFAULT),
                    help="comma-separated lattice indices k; lambda = k * 5.5/14, i.e. Phase 3's zoom "
                         "grid. k <= 14 pairs exactly with an existing fine_decoded_max5.5 image; k > 14 "
                         "extends past anything Phase 3 ran. (default: %(default)s)")
    ap.add_argument("--lam-step", type=float, default=LAM_STEP_PHASE3,
                    help="lattice spacing; lambda = k * step. Default %(default)g is Phase 3's 5.5/14, "
                         f"which pairs with its stored images. The prompt study uses "
                         f"{LAM_STEP_PROMPT_STUDY} with k=1..5 (lambda 0.2..1.0). The step enters the "
                         "stamp, so two different lattices never merge into one results file.")
    ap.add_argument("--sweep-seed", type=int, default=SWEEP_SEED,
                    help="seed threaded through the sampler (z0 AND corrector noise, unless --z0-seed "
                         "is given). Default %(default)s is Phase 3's, so z0 is IDENTICAL to its sweep.")
    ap.add_argument("--z0-seed", type=int, default=None,
                    help="if set, draw z0 from THIS seed and let --sweep-seed vary only the corrector "
                         "noise. Running 3 jobs with the same --z0-seed and different --sweep-seed "
                         "measures PC's within-seed (Langevin) variance, which Phase 3 structurally does "
                         "not have and which is what sizes the seed replication.")
    ap.add_argument("--corrector-steps", type=int, default=CORRECTOR_STEPS,
                    help="ULA steps per ODE step (default: %(default)s). Ignored by the flow_guided arm.")
    ap.add_argument("--snr", type=float, default=SNR,
                    help="Langevin SNR (default: %(default)s, Song et al. 2021).")
    ap.add_argument("--eta-reference", choices=("total", "score"), default=ETA_REFERENCE,
                    help="'total' (default) divides eta by ||s + lam*g~||, which anneals the corrector's "
                         "step as ~1/(1+lam) -- measured 12x weaker at lam=5.5 than at lam=0.79. 'score' "
                         "divides by ||s|| alone, so eta is lam-independent and lam only ROTATES the "
                         "drift; use it to tell a real high-lam null apart from the corrector having "
                         "annealed itself off.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the tier-1 on-hardware checks (~4 min). Only for a resumed job whose "
                         "earlier attempt already logged them all PASS.")
    args = ap.parse_args()

    CORRECTOR_STEPS = args.corrector_steps
    SNR = args.snr
    ETA_REFERENCE = args.eta_reference
    N_STEPS = args.n_steps
    LAM_STEP = args.lam_step
    lam_k = [int(k) for k in args.lam_k.split(",") if k.strip() != ""]
    LAM_GRID = [k * LAM_STEP for k in lam_k]

    arms = [a.strip() for a in args.arm.split(",") if a.strip() != ""]
    unknown = [a for a in arms if a not in ARMS]
    if unknown or not arms:
        ap.error(f"--arm must be a comma-separated subset of {ARMS}; got {args.arm!r}")

    S = dry_setup(args.prompt) if args.dry_run else flux_setup(args.prompt)
    note(f"setup ready: arms={arms} prompt={args.prompt!r} d={S.d} device={S.device} "
         f"lambda_s={S.lam_s:.2f} n_steps={N_STEPS} corr_steps={CORRECTOR_STEPS} snr={SNR} "
         f"eta_ref={ETA_REFERENCE} lam_step={LAM_STEP:g} lam_k={lam_k} lam_grid={LAM_GRID} "
         f"z0_seed={args.z0_seed}")

    # ONE preflight for the whole job, not one per arm: every item it checks (GPU model, reward
    # normalization, lambda_s, the score reparametrization, and the corrector_steps=0 reduction) is a
    # property of the SETUP, which all arms in this job share by construction.
    pf: dict[str, Any] = {}
    if not args.skip_preflight:
        pf = preflight(S, args.sweep_seed, dry_run=args.dry_run, prompt=args.prompt)

    for arm in arms:
        # The results key encodes EVERY axis a wave varies. Without this, two jobs differing only in
        # --corrector-steps or --sweep-seed write to the same file: the stamp check would .stale-rename
        # rather than corrupt, but each job would keep discarding the other's work. Same lesson as
        # CLAUDE.md's "bump RUN_TAG per leg -- an untagged leg overwrites its own resume source".
        # The prompt is in here for the same reason, and it is load-bearing: a prompt is a different
        # reward, so two prompts sharing a file would silently .stale-stomp each other every job.
        key = (f"{prompt_slug(args.prompt)}_{arm}_n{N_STEPS}_c{CORRECTOR_STEPS}_eta{ETA_REFERENCE}"
               f"_snr{SNR:g}_s{args.sweep_seed}"
               + (f"_z{args.z0_seed}" if args.z0_seed is not None else "")
               + (f"_{args.tag}" if args.tag else "")
               + (".dryrun" if args.dry_run else ""))
        RESULTS = os.path.join(HERE, f"pc_sweep_results_{key}.json")
        DECODED_DIR = os.path.join(HERE, f"pc_decoded_{key}")
        if args.dry_run and os.path.exists(RESULTS):
            os.remove(RESULTS)
        note(f"[{arm}] results -> {os.path.basename(RESULTS)}")

        res = load_results({
            **S.stamp, "arm": arm, "n_particles": N_PARTICLES, "n_steps_ode": N_STEPS,
            "lambda_s": S.lam_s, "lam_step": LAM_STEP, "lam_k": lam_k, "lam_grid": LAM_GRID,
            "sweep_seed": args.sweep_seed, "z0_seed": args.z0_seed,
            "corrector_steps": CORRECTOR_STEPS, "snr": SNR,
            "eta_reference": ETA_REFERENCE, "exact_jacobian": EXACT_JACOBIAN,
            "preflight": {k: v for k, v in pf.items() if k != "score_xval_s"},
        })
        for k, lam in zip(lam_k, LAM_GRID):
            run_one(S, arm, k, lam, res, args.sweep_seed, args.z0_seed)

        print(f"\nprompt={args.prompt!r}  arm={arm}  corr_steps={CORRECTOR_STEPS}  snr={SNR}  "
              f"eta_ref={ETA_REFERENCE}")
        print(f"{'idx':>4s} {'m':>7s} {'lam':>8s} {'f_mean':>12s} {'|x|/sqrt d':>11s} {'hf_frac':>8s} "
              f"{'eta':>10s} {'disp':>8s} {'noisefrac':>10s} {'t(s)':>8s} {'oom':>4s}")
        for k, lam in zip(lam_k, LAM_GRID):
            pkey = f"idx{k:02d}_lam{lam:.4f}"
            if pkey not in res:
                print(f"{k:>4d}  (missing)")
                continue
            r = res[pkey]
            print(f"{k:>4d} {r['m']:>7.3f} {r['lam']:>8.2f} {r['f_mean']:>12.4f} "
                  f"{r['x_norm_final']:>11.4f} {r['hf_frac']:>8.4f} {r['corrector_eta_mean']:>10.3g} "
                  f"{r['corrector_rel_disp_mean']:>8.3g} {r['corrector_noise_frac_mean']:>10.3f} "
                  f"{r['t_sample_s']:>8.1f} {'Y' if r['oom_fallback_any'] else '-':>4s}")

    note("done")


if __name__ == "__main__":
    main()
