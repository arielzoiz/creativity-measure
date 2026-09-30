# Diamond Maps: does the tilt actually raise E_q[f]?

Follow-up to `notebooks/diamond_smc_imagenet/`. That directory's real run (job 698463,
`results_label207_cfg1.0_M16K4N6.pt`) already answers this more precisely than the grid image alone
lets on -- its cell 7 selection table (recovered from `run_698463.ipynb`'s saved outputs) is the
starting point for everything here:

```
    m    lambda    E_q[f]  min ESS/M   uniq/M    sec  verdict
  0.0     0.000    0.9965       1.00     1.00     39  OK
  1.0    29.114    1.0014       0.66     0.56     30  OK        <- selected
  2.0    58.227    1.0209       0.31     0.38     30  DEGENERATE
  3.0    87.341    1.0208       0.12     0.19     33  DEGENERATE
```

At the one point the established 0.5-diversity-floor rule actually allows (m=1.0), the tilt buys
**+0.005** over the untilted control -- essentially nothing, against `E_p[f]=1` by construction. The
larger-looking gains at m=2/3 (+0.02) only appear once `uniq/M` has already collapsed to 0.38/0.19 --
i.e. only a handful of surviving lineages are left to report a number, which is exactly the failure
mode the 0.5 floor exists to catch. So: **no, this run is not achieving meaningfully higher `E_q[f]`**,
independent of the separate image-blur problem (`notebooks/diamond_smc_step_regression/`).

## Two questions, two concurrent jobs

- **`m_sweep.py`**: re-runs the identical `p` (label=207, cfg_scale=1.0, N=6, K=4, M=16) at a finer
  `m` grid (`{0, 0.25, ..., 3.0}`, 11 points) than the original `{0,1,2,3}`, to locate the OK/DEGENERATE
  crossing precisely and see whether `E_q[f]` rises smoothly as diversity falls or jumps.
- **`M_sweep.py`**: holds `lambda` FIXED at the m=2.0 value from job 698463 (58.227) and sweeps
  particle count `M in {16, 32, 64, 128}`, testing whether more particles rescues `min ESS/M` /
  `uniq/M` back over the 0.5 floor at that same tilt strength -- mirroring Algorithm 3's established
  `ESS/M ~ exp(-m^2)` scaling (CLAUDE.md), which would predict the fix is `M`, not `lambda`, since
  Algorithm 2 is also SMC-with-resampling over a fixed trajectory rather than MCMC that moves
  particles (Algorithm 1's pCN kernel).

Both reuse `reward_common.py` for backend/reward construction (byte-identical to
`diamond_smc_imagenet.ipynb`'s own setup, so results are directly comparable) and are otherwise
independent -- no shared state, so they run as two separate Slurm jobs concurrently.

## Running

```sh
sbatch m_sweep.slurm
sbatch M_sweep.slurm
```

Budget ~30-60 min each, dominated by one checkpoint unpickling per job (~10-20 min on this cluster's
storage) plus per-run SMC time (~30-40s/run at M=16 per job 698463's own timings; larger at M=64/128).

## Takeaways

**Neither experiment found a way to make the tilt do more than it already does.** Both ran to
completion (`m_sweep` twice, `M_sweep` partially -- see below), on `label=207, cfg_scale=1.0, N=6,
K=4`, matching job 698463 exactly.

**1. `m_sweep`: the OK region is flat-to-tiny, and the OK/DEGENERATE transition is not reproducible.**
Two independent runs (same seed, same everything) of the fine grid `m in {0, 0.25, ..., 3.0}`:

```
  run 1 (905870)                         run 2 (905954, with images -> m_sweep_grid.png)
    m   E_q[f]  DeltaE_q[f]  verdict        m   E_q[f]  DeltaE_q[f]  verdict
 0.00  0.9964    +0.0000    OK          0.00  0.9965    +0.0000    OK
 0.25  0.9948    -0.0016    OK          0.25  0.9948    -0.0017    OK
 0.50  0.9881    -0.0084    OK          0.50  0.9934    -0.0030    OK
 0.75  1.0001    +0.0037    OK          0.75  0.9925    -0.0040    OK
 1.00  1.0020    +0.0055    OK          1.00  1.0016    +0.0051    OK
 1.25  1.0014    +0.0050    OK          1.25  1.0096    +0.0131    OK
 1.50  1.0548    +0.0584    DEGENERATE  1.50  0.9943    -0.0022    DEGENERATE
 1.75  1.0256    +0.0291    DEGENERATE  1.75  1.0263    +0.0299    DEGENERATE
 2.00  1.0202    +0.0238    DEGENERATE  2.00  1.0169    +0.0204    DEGENERATE
 2.50  1.0150    +0.0186    DEGENERATE  2.50  1.0153    +0.0188    DEGENERATE
 3.00  1.0261    +0.0296    DEGENERATE  3.00  1.0289    +0.0324    DEGENERATE
```

The OK region (`m<=1.25`, both runs) tops out at `DeltaE_q[f] ~ +0.005` to `+0.013` -- essentially
nothing against `E_p[f]=1`. The two runs AGREE on that. They DISAGREE sharply at the OK/DEGENERATE
boundary itself: run 1 spikes to +0.058 exactly at `m=1.50`; run 2 barely moves (-0.002) at the same
`m`. **Degenerate-region readings are not systematically inflated -- they are just noise, and the
sign of that noise is not reproducible run to run.** (Revises the earlier read of job 698463's coarser
grid, which only had one run to look at and so couldn't distinguish "inflated" from "noisy.")

**2. The images explain why: the tilt is reshuffling a small fixed menu, not generating anything.**
`m_sweep_grid.png` (run 2, decoded images per m) shows entire columns frozen bit-for-bit across long
stretches of `m` -- column 1 reads `f=1.019` in EVERY row from `m=0.00` to `m=2.50` (finally changes at
`m=3.00`), and every one of the 6 shown images is identical between `m=1.00` and `m=1.25` despite a
25% jump in `lambda`. This is expected given the sampler's own design (the base RNG stream is held
fixed across an `m`-sweep so only `lambda` varies, per `diamond_smc.py`'s docstring) -- but it makes
concrete exactly what "the tilt does almost nothing" means mechanically: at `M=16`, there are only
`~16` base-trajectory outcomes per run for resampling to redistribute weight over, and most of the
`m`-sweep just keeps picking the same handful of them.

**3. `M_sweep`: more particles did NOT rescue the degeneracy -- if anything it got worse.** Held
`lambda=58.227` (job 698463's m=2.0) fixed, swept `M`:

```
   M   E_q[f]  min ESS/M   uniq/M
  16   1.0141     0.19      0.31
  32   1.0316     0.24      0.25
  64   1.0341     0.20      0.16
 128   OOM (reward call batches M*K*NUM_EPS = 128*4*3 = 1536 rows through the score network;
        XLA_PYTHON_CLIENT_MEM_FRACTION=.5 caps JAX to half the 46 GB card)
```

`uniq/M` fell (0.31 -> 0.25 -> 0.16) as `M` rose from 16 to 64 -- the opposite of Algorithm 3's
established `M ~ exp(m^2)` escape hatch (CLAUDE.md). Whatever is capping diversity at this `lambda`,
it is not simply "not enough particles," and the fix that works for Algorithm 3 does not transfer here.
`M=128` was not retried (would need the reward call batched to dodge the OOM) since the trend was
already the wrong direction.

**4. Why the base images are blurry in the first place is now understood, and it is not a bug in this
project.** The Figure 18 caption you're benchmarking against reads "guidance with Posterior Diamond
Maps" -- that is the paper's OTHER algorithm (a single deterministic Euler step per outer step, only
nudged by a reward gradient -- never re-draws stochastic noise), not SMC. Checked directly against the
upstream repo: `configs/imagenet/smc.py` (the paper's own released SMC config, which
`diamond_maps_jax.py` replicates exactly) specifies `SiT-XL-2.pkl` with `use_glass=True` for the base
trajectory -- i.e. the paper's own prescribed SMC-on-ImageNet setup requires GLASS's stochastic
outer transitions, the same mechanism `notebooks/diamond_smc_step_regression/` showed compounds a
variance leak with every additional step. The paper never publishes an SMC qualitative result at this
resolution to compare against; its own `configs/celeba64/` doesn't even ship an `smc.py`. There is no
FLUX-equivalent general text-to-image checkpoint in `MonkeyDoug/diamond-maps` (only ImageNet and
CelebA), and while a deterministic `MeanFlow-XL-2.pkl` checkpoint exists for ImageNet, wiring it in as
SMC's base generator isn't supported by the released config and would remove the per-step stochasticity
SMC's resampling needs to have anything to select among in the first place -- at which point it stops
being SMC and starts looking like Algorithm 3.

**Net picture.** On this backend (SiT-XL-2 + GLASS + ImageNet-DiamondMap-B2, as the paper's own
released SMC config specifies it), Algorithm 2 combines (a) a base generator that gets blurrier with
every additional outer step by a mechanism unrelated to the reward, and (b) a reward tilt that, even at
its best trustworthy setting, moves `E_q[f]` by an amount comparable to run-to-run noise -- because
resampling only ever reshuffles a small, fixed pool of base-trajectory outcomes, and neither `lambda`
nor `M` widen that pool. Neither problem looks fixable by tuning knobs already exposed by this
backend/config.
