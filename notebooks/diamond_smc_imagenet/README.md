# Diamond Maps SMC (Algorithm 2) with the normalized squared global IEM tilt

Everything needed to *run* the experiment. The sampler itself is library code
(`creativity_measure/diamond_smc.py` + `creativity_measure/backends/diamond_maps_jax.py`); nothing
in this directory is imported by the library.

## Files

| file | what it is |
|---|---|
| `diamond_smc_imagenet.ipynb` | the run. All knobs — including `LABEL` and `CFG_SCALE`, which define `p` — are in the first code cell. |
| `diamond_smc.slurm` | submits the notebook via `nbconvert` to a fresh output notebook `run_<jobid>.ipynb`. |
| `smoke_test.py` | the validation ladder at throwaway settings: rungs 1-6, plus a tiny end-to-end run. |
| `smoke_test.slurm` | submits `smoke_test.py`. **Run this first.** |
| `paths.py` | resolves the checkpoints in the HF cache and the upstream clone; GPU/asset preflight. |

## Checkpoints are not stored here

They live in the HuggingFace cache (`$HF_HOME` = `$WORK/.cache/huggingface`, redirected off the 2 GB
home in `~/.cshrc`) and are addressed by path — `DiamondMapsBackend` overrides upstream's
`repo_paths.ckpt_path` default at config time. Nothing is copied into the project, and the
`diamond_maps` clone stays a pristine checkout that can be `git pull`ed.

Download once, from a login node (10.7 GB, ~2 min):

```sh
hf download MonkeyDoug/diamond-maps \
    --include 'ckpt/sit_assets/SiT-XL-2.pkl' 'ckpt/ImageNet-DiamondMap-B2.pkl'
```

Two checkpoints, three jobs:

- **`SiT-XL-2.pkl`** (8.1 GB) — base network: the DDPM transitions (Alg. 2 line 6, via GLASS) **and**
  the marginal score for the IEM reward.
- **`ImageNet-DiamondMap-B2.pkl`** (2.6 GB) — posterior network: the one-NFE lookahead (line 9).

`SiT-B-2.pkl` is *not* needed — it belongs to upstream's flow-posterior baseline variant.

## Environment

Conda env `diamond-creative` (Python 3.11), exact versions in `environment.txt`. The load-bearing
pins, each of which broke something when left free:

| pin | why |
|---|---|
| `jax[cuda12]==0.4.30`, `flax==0.8.2`, `optax==0.2.2` | upstream's own pins; the checkpoints are Flax state dicts |
| `torch==2.4.1` | upstream's pin. Only the reward needs torch, but see the two rows below |
| `transformers==4.57.6` | **unpinned resolves to 5.x, which imports `DTensor` from `torch.distributed.tensor` — a path that only exists in torch ≥ 2.5.** `diffusers` imports transformers transitively while loading the Flax VAE, so the VAE decode dies with an unrelated-looking `ImportError` |
| `diffusers==0.36.0` | upstream's pin; still ships `FlaxAutoencoderKL` (deprecated, removed in v1.0.0) |
| `tensorflow-cpu==2.15.0` | caps Python at 3.11. CPU build on purpose: TF is pulled in only by upstream's module-level imports and would otherwise compete with jax and torch for VRAM |
| `tensorflow-metadata==1.14.0` | pinned back — the current release needs a protobuf that TF 2.15 forbids |

Plus `creativity-measure` installed editable.

## Running

```sh
sbatch smoke_test.slurm                     # plumbing, ~minutes. Do this first.
sbatch diamond_smc.slurm                    # the run, with the notebook's defaults

# override knobs without editing the notebook:
sbatch --export=ALL,DMSMC_LABEL=151,DMSMC_LAM_MULTS=0,1,2,4 diamond_smc.slurm
```

Both scripts set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=.5`,
without which jax grabs ~75% of VRAM on first use and torch then OOMs inside the reward.

Never on the login node: its TITAN Xps are Pascal (no native bf16) and the configs compute in bf16.

## Cost

Per λ: ~10.8k SiT-XL/2 single-image passes ≈ 1.3 PFLOP ≈ tens of seconds of arithmetic on an
L40S/A6000. Wall time is dominated by startup — unpickling 10.7 GB and XLA-compiling the two steps —
so budget ~5-10 min for the first λ and ~1 min for each additional λ in the same process.

The reward is ~88% of per-step cost. `R` is *not* a cost lever: the reference score bank is cached
(`global_iem.py`), so per-call cost is `(N_GAMMA-1)·NUM_EPS·B`, independent of `R`. To cut cost,
lower `K` first, then `N_GAMMA` — but note `N_GAMMA` is a systematic quadrature bias in the metric,
not noise, so it does not average out.
