"""Shared setup for the auto-R diagnostic, which runs as TWO parallel Slurm jobs.

`refset_auto_r_diagnostic.ipynb` (RandomRefs) and `refset_auto_r_fps.ipynb` (WeightedFPSRefs) ask the
same question -- how many IEM references does f need before its RANKING of samples stops moving? --
about two selectors that fail differently. Their tau curves are only comparable if both arms see the
IDENTICAL f: same model, prompt, guidance, gamma grid, Brownian seed, denoiser chunking and probe
points. Copy-pasting that setup into two notebooks is how it silently drifts, so it lives here.

Everything below is pinned to `notebooks/flux_lambda_sweep_strong_2` (jobs 695266/695271/695455), so
the answer applies to THAT reward rather than a cheaper, better-conditioned one.

WHAT CHANGED SINCE THE FAILED RUN (job 694256, N_GAMMA=6, R<=32, both arms OOM):

1. `chunked_denoiser` (edm_adapter) bounds the transformer batch, so the notebook no longer needs its
   own row-chunked Distance subclass. The OOM that killed both arms is fixed in the library.
2. `GlobalIEMDistance` caches the reference score bank, so a point's score is evaluated ONCE per
   (x, gamma, W) instead of once per comparison. In the SMC that makes the per-sweep cost independent
   of R. It does NOT make R free HERE: this diagnostic's whole cost IS building those banks, and that
   is linear in the number of distinct points scored. Budget the run in POINTS BANKED, not in R.

COST MODEL (measured, not assumed). Per point:
    generation  2*N_STEPS - 1     = 15 transformer rows   (Heun: 2/step, minus the last corrector)
    score bank  (N_GAMMA-1)*NUM_EPS = 87 rows             (one score per gamma interval per path)
at S_ROW = 0.22 s/row -- job 695271's R=64 reference bank took 1404 s for 6528 rows = 0.215 s/row, and
job 695266 measured 0.25-0.26 s/row across batch widths 24..128. So ~22 s per point banked.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Generator, Tensor

from creativity_measure import SquaredGlobalIEMDistance, set_default_dtype
from creativity_measure.distances.edm_adapter import chunked_denoiser, edm_score_fn
from creativity_measure.distances.global_iem import _scores_at_gamma, iem_sq_integral
from creativity_measure.distances.utils import simulate_brownian
from creativity_measure.generators.base import edm_generator

# --- the sweep's configuration, copied verbatim -----------------------------------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "A dog"
GUIDANCE = 1.5                      # run 2's value (the baseline sweep used 3.5)
IMG = 512
C, H, W = 16, IMG // 8, IMG // 8
LATENT_DIM = C * H * W              # 65536
SIGMA_MIN, SIGMA_MAX = 2e-3, 80.0
N_STEPS = 8
MAX_DENOISER_ROWS = 24              # measured ceiling 128, OOM at 192 (job 695266); s/row is flat
N_GAMMA, NUM_EPS = 30, 3
DIST_SEED = 123                     # Brownian bank seed, as in the sweep

# --- diagnostic controls shared by both arms --------------------------------------------------------
SEED = 17                           # selector seed; both arms use it, so both probe the SAME points
PROBE_SIZE = 128                    # tau resolution: SE ~ sqrt(2(2n+5)/(9n(n-1))) = 0.060 at n=128
TAU_TARGET = 0.9                    # nominal bar; both arms pass fallback="best" so a lower ceiling
                                    # reports instead of raising -- the ceiling is itself a finding
S_ROW = 0.22                        # s per transformer row (see module docstring)
GEN_ROWS = 2 * N_STEPS - 1          # 15
BANK_ROWS = (N_GAMMA - 1) * NUM_EPS  # 87

# VRAM: one bank row is (N_GAMMA-1)*NUM_EPS*d floats = 22.8 MB per point.
BANK_BYTES_PER_POINT = BANK_ROWS * LATENT_DIM * 4


def hours(rows: int) -> float:
    """Projected wall-clock hours for `rows` transformer rows."""
    return rows * S_ROW / 3600.0


def check_model_access() -> str:
    """Confirm FLUX.1-dev is reachable, WITHOUT assuming a credential is readable on this node.

    FLUX.1-dev is gated, so the sweeps have always preflighted with `auth_check`, which needs the HF
    token. The token lives in $HOME (/a/home/cc/...) because $WORK is group-readable -- and $HOME is
    not mounted on every node in `killable`: job 697403 died in 33 s with a 401 on t-806 after the
    identical setup had worked on n-805 an hour earlier.

    Nothing here needs the network: the full 32 GB snapshot is already in HF_HOME under $WORK. So when
    the token is absent the caller sets HF_HUB_OFFLINE=1 and this validates the local snapshot instead
    -- a stronger check than auth_check, since it verifies the bytes rather than the permission.
    """
    from huggingface_hub import auth_check, snapshot_download

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        path = snapshot_download(MODEL_ID, local_files_only=True)   # raises if anything is missing
        print(f"FLUX.1-dev: offline, complete local snapshot at {path}")
        return path
    auth_check(MODEL_ID)
    print("FLUX.1-dev access: OK (authenticated)")
    return ""


# =====================================================================================================
# Sampler
# =====================================================================================================


def points_fingerprint(x: Tensor) -> list[float]:
    """Cheap content signature of a point set, to validate a checkpoint without hashing 67 MB.

    Two different draws agreeing on all four of these is not a scenario worth engineering against;
    the realistic failure it must catch is a checkpoint left over from a DIFFERENT configuration,
    which changes every one of them.
    """
    flat = x.reshape(-1)
    return [float(x.sum()), float(x.square().sum()), float(flat[0]), float(flat[-1])]


def atomic_save(obj: object, path: str) -> None:
    """Write via a temp file, so a preemption mid-write cannot leave a half-file that loads as valid."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class MemoSampler:
    """G(z) behind the `.sample(n, seed=...)` interface RefSelector expects, memoized by (n, seed).

    Memoization is not (only) about saving the 15 generation rows. It is what makes the score-bank
    cache below actually HIT: the selectors re-request the same point sets (the probes, the FPS pool,
    each estimation set) at several places in one auto-R sweep, and a cache keyed by tensor identity
    can only recognise them if the same request returns the SAME tensor object.

    Deliberate consequence across the two arms: both use seed SEED, so both draw probes from
    SEED + 9973 and therefore rank the identical probe points -- their tau values are comparable
    directly, not merely in distribution.

    `cache_dir` extends the memo to disk. G is deterministic given (n, seed) and the model config, so
    a restart regenerates identical points -- but at 15 rows each, re-deriving 256 references costs
    ~11 min. On a preemptible partition that is paid on every requeue, so it is cached with a config
    stamp that invalidates it if anything about G changes.
    """

    def __init__(self, G, d: int, device: torch.device, cache_dir: str | None = None) -> None:
        self.G = G
        self.d = d
        self.device = device
        self.cache_dir = cache_dir
        self._cache: dict[tuple[int, int], Tensor] = {}
        self.rows_generated = 0
        self.disk_hits = 0

    def _path(self, key: tuple[int, int]) -> str | None:
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"sample_n{key[0]}_seed{key[1]}.pt")

    def sample(
        self, n: int, seed: int | None = None, *, generator: Generator | None = None
    ) -> Float[Tensor, "n d"]:
        key = (n, 0 if seed is None else int(seed))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        path = self._path(key)
        stamp = {"guidance": GUIDANCE, "n_steps": N_STEPS, "sigma_min": SIGMA_MIN,
                 "sigma_max": SIGMA_MAX, "d": self.d, "img": IMG, "prompt": PROMPT}
        if path is not None and os.path.exists(path):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("stamp") == stamp and tuple(blob["x"].shape) == (n, self.d):
                x = blob["x"].to(self.device)
                self._cache[key] = x
                self.disk_hits += 1
                print(f"  [ckpt] reused {n} points (seed {key[1]}) from disk")
                return x
            print(f"  [ckpt] stale sample checkpoint {os.path.basename(path)} ignored "
                  f"(configuration changed)")
        gen = torch.Generator(device="cpu").manual_seed(key[1])
        with torch.no_grad():
            x = self.G(torch.randn(n, self.d, generator=gen).to(self.device))
        self.rows_generated += n * GEN_ROWS
        self._cache[key] = x
        if path is not None:
            atomic_save({"x": x.cpu(), "stamp": stamp}, path)
        return x


# =====================================================================================================
# Distance with a two-sided, budgeted score-bank cache
# =====================================================================================================


class CachedSquaredGlobalIEM(SquaredGlobalIEMDistance):
    """`SquaredGlobalIEMDistance` whose score banks are cached for BOTH arguments, under a byte budget.

    WHY. The library caches the bank of `x_refs` only, in a single slot -- exactly right for the SMC,
    where one frozen reference set is compared against every batch. The auto-R sweep has the opposite
    access pattern: the same LEFT argument recurs while the right one changes every call.
    `FPSRefs._ensure_order` is the extreme case -- it calls `pairwise(pool, one_new_ref)` once per
    greedy pick, so an uncached pool is re-scored on every pick: at pool=160 that is 87*160 rows per
    pick, ~48 h for 64 picks. Cached, the pool is banked once and each pick costs 87 rows. The FPS arm
    is not slow without this; it is impossible.

    EXACTNESS. A bank depends only on (points, gammas, num_eps, seed) -- never on what the points are
    compared against (`global_iem.score_bank`). Serving it from cache reproduces the same tensor the
    recompute would have produced, bit-for-bit: identical inputs, identical per-gamma calls, identical
    chunking (`MAX_DENOISER_ROWS` is fixed and the row count per call, num_eps*P, depends only on P).

    KEYS AND LIFETIME. Keyed by `_points_key` (data_ptr + shape + dtype + device + num_eps + seed +
    gammas identity). A data_ptr is only unique among LIVE tensors, so each entry holds a reference to
    its points tensor: while the entry exists the address cannot be recycled under it. Eviction is LRU
    against `cache_bytes`, and room is made BEFORE a build rather than after, so the budget bounds the
    true peak instead of being exceeded by exactly the largest bank at the worst moment.
    """

    def __init__(self, *args, cache_bytes: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_bytes = int(cache_bytes)
        self._banks: OrderedDict[tuple, tuple[Tensor, Tensor]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.rows_scored = 0
        self.evictions = 0

    # ---- cache ---------------------------------------------------------------------------------

    def cached_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for _, b in self._banks.values())

    def _make_room(self, need: int) -> None:
        while self._banks and self.cached_bytes() + need > self.cache_bytes:
            self._banks.popitem(last=False)          # LRU
            self.evictions += 1
        if self.evictions and torch.cuda.is_available():
            torch.cuda.empty_cache()                 # return the freed blocks to the allocator

    def _build_bank(
        self,
        points: Float[Tensor, "P d"],
        W: Float[Tensor, "N_gamma N_eps 1 d"],
        gammas: Float[Tensor, "N_gamma"],
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps P d"]:
        """`global_iem.score_bank`, but writing into a preallocated output.

        `score_bank` builds a list of the N_gamma-1 per-gamma slices and then `torch.stack`s it, so
        the slices and their concatenation are both resident at the peak -- 2x the bank. At P=256 that
        is 10.9 GB instead of 5.4 GB, next to a 22.3 GB transformer on a 48 GB card. Filling a
        preallocated buffer holds one slice (200 MB) extra instead. Same values, same order.
        """
        n_int = gammas.shape[0] - 1
        out = points.new_empty((n_int, self.num_eps, points.shape[0], points.shape[1]))
        for i in range(n_int):
            out[i] = _scores_at_gamma(points, gammas[i], W[i], self.density, self.score_fn)
        self.rows_scored += n_int * self.num_eps * points.shape[0]
        return out

    def bank(
        self,
        points: Float[Tensor, "P d"],
        W: Float[Tensor, "N_gamma N_eps 1 d"],
        gammas: Float[Tensor, "N_gamma"],
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps P d"]:
        key = self._points_key(points)
        got = self._banks.pop(key, None)
        if got is not None:
            self._banks[key] = got                   # LRU touch
            self.hits += 1
            return got[1]
        self.misses += 1
        self._make_room((gammas.shape[0] - 1) * self.num_eps
                        * points.shape[0] * points.shape[1] * points.element_size())
        bank = self._build_bank(points, W, gammas)
        self._banks[key] = (points, bank)             # holds `points` alive -> the data_ptr key is safe
        return bank

    def retain(self, keep: list[Tensor]) -> None:
        """Drop every cached bank except those of `keep`. Called between phases to hand the next one a
        cache holding exactly what it will reuse, instead of trusting LRU to guess."""
        wanted = {self._points_key(t) for t in keep}
        for key in [k for k in self._banks if k not in wanted]:
            del self._banks[key]
            self.evictions += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- checkpointing (killable preempts; job 698172 lost 33 min to it) -------------------------

    def _bank_stamp(self, points: Tensor) -> dict:
        """Everything a bank's values depend on. A checkpoint whose stamp differs is not this bank."""
        return {"points": points_fingerprint(points), "shape": tuple(points.shape),
                "n_gamma": int(self.gammas.shape[0]), "num_eps": self.num_eps, "seed": self.seed,
                "gammas": [float(self.gammas[0]), float(self.gammas[-1])],
                "dtype": str(points.dtype)}

    def save_bank(self, points: Tensor, path: str) -> bool:
        """Persist `points`' cached bank. Returns False if it was not in the cache to begin with."""
        got = self._banks.get(self._points_key(points))
        if got is None:
            return False
        atomic_save({"bank": got[1].cpu(), "stamp": self._bank_stamp(points)}, path)
        return True

    def load_bank(self, points: Tensor, path: str) -> bool:
        """Restore a saved bank into the cache under `points`' key. False if absent or stale.

        The restored tensor is the one the rebuild would have produced -- the stamp covers every input
        a bank depends on -- so this is a pure time saving, not an approximation.
        """
        if not os.path.exists(path):
            return False
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("stamp") != self._bank_stamp(points):
            print(f"  [ckpt] stale bank checkpoint {os.path.basename(path)} ignored")
            return False
        bank = blob["bank"]
        self._make_room(bank.numel() * bank.element_size())
        self._banks[self._points_key(points)] = (points, bank.to(points.device))
        return True

    def stats(self) -> str:
        return (f"bank cache: {self.hits} hits / {self.misses} misses, {self.evictions} evictions, "
                f"{self.cached_bytes() / 2**30:.2f} GB resident of {self.cache_bytes / 2**30:.1f} GB, "
                f"{self.rows_scored:,} rows scored")

    # ---- pairwise ------------------------------------------------------------------------------

    def pairwise(
        self, X: Float[Tensor, "B d"], x_refs: Float[Tensor, "R d"]
    ) -> Float[Tensor, "B R"]:
        """`GlobalIEMDistance.pairwise`, with BOTH banks served from the LRU cache.

        The only deviation from the library is where the two `score_bank`s come from; the Brownian
        bank, the integral and the final reduction are the parent's.
        """
        device, dtype = X.device, X.dtype
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = simulate_brownian(gammas, self.num_eps, X.shape[1], self.seed, device, dtype)
        ref_scores = self.bank(x_refs, W, gammas)
        same = self._points_key(X) == self._points_key(x_refs)
        batch_scores = ref_scores if same else self.bank(X, W, gammas)
        iem_sq = iem_sq_integral(
            x_refs, X, W, gammas, self.density, self.score_fn,
            ref_scores=ref_scores, batch_scores=batch_scores,
            r_chunk=self.r_chunk, verbose=self.verbose,
        )
        return self._finalize(iem_sq.mean(0).clamp_min(0))


# =====================================================================================================
# Setup
# =====================================================================================================


@dataclass
class Setup:
    device: torch.device
    p_gen: MemoSampler
    distance: CachedSquaredGlobalIEM
    gammas: Tensor
    data_scale: float
    notes: list[str] = field(default_factory=list)

    def report(self) -> None:
        for line in self.notes:
            print(line)


def build(cache_reserve_gb: float = 8.0, verbose: bool = True) -> Setup:
    """Load FLUX, build G / score_fn / the cached distance, and size the bank cache to the card.

    cache_reserve_gb: VRAM held back from the cache for transformer activations and allocator slack.
        The measured working set at MAX_DENOISER_ROWS=24 is ~3.6 GB (job 695266: 27.3 GB peak with the
        model's 22.3 GB and a 1.36 GB bank resident), so 8 GB is that plus margin for fragmentation.
    """
    set_default_dtype(torch.float32)          # FLUX runs bf16; everything the metric touches is fp32
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    notes: list[str] = []

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    # Free both text encoders (~9.5 GB, T5 dominates): the embeddings above are all we need.
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transformer = pipe.transformer.eval()

    img_ids = FluxPipeline._prepare_latent_image_ids(1, H // 2, W // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        """EDM denoiser D(x_sigma, sigma) = E[X | x_sigma] backed by the FLUX transformer.

        Flow matching interpolates, EDM adds: x_t = (1-t)x_0 + t*eps vs x_sigma = x_0 + sigma*eps.
        Factoring out (1-t) makes them identical under t = sigma/(1+sigma), x_t = x_sigma/(1+sigma).
        FLUX predicts v = eps - x_0, so x_t - t*v = x_0 exactly. Byte-identical to the sweep's.
        """
        b = x.shape[0]
        sig = sigma.reshape(b, 1, 1, 1).to(torch.float32)
        t = (sigma / (1.0 + sigma)).to(torch.float32)
        x_t = x.to(torch.float32) / (1.0 + sig)
        with torch.no_grad():
            v = transformer(
                hidden_states=FluxPipeline._pack_latents(x_t.to(dtype), b, C, H, W),
                timestep=t.to(dtype),
                guidance=torch.full((b,), GUIDANCE, device=x.device, dtype=torch.float32),
                pooled_projections=pooled_prompt_embeds.expand(b, -1).to(dtype),
                encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(dtype),
                txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
        v = FluxPipeline._unpack_latents(v, IMG, IMG, 8).to(torch.float32)
        return (x_t - t.reshape(b, 1, 1, 1) * v).to(torch.float32)

    # Wrap ONCE and build both consumers from the wrapper, so the generator path and every score path
    # -- including the widest call in the run, the bank build -- share one batch ceiling.
    denoiser_capped = chunked_denoiser(denoiser, MAX_DENOISER_ROWS)
    G = edm_generator(denoiser_capped, img_shape=(C, H, W),
                      sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, n_steps=N_STEPS)
    score_fn = edm_score_fn(denoiser_capped, (C, H, W))
    p_gen = MemoSampler(G, LATENT_DIM, device)

    # gamma grid: GAMMA_LO = 1/S^2 from the data length scale (beyond sigma = S the observation is pure
    # noise). S is estimated from the probe set itself -- the same points both arms rank.
    probes = p_gen.sample(PROBE_SIZE, seed=SEED + 9973)
    S = float(probes.std())
    gamma_lo = max(1.0 / S**2, 1.0 / SIGMA_MAX**2)
    gamma_hi = min(2.0**10, 1.0 / SIGMA_MIN**2)
    gammas = torch.logspace(math.log2(gamma_lo), math.log2(gamma_hi), N_GAMMA, base=2).to(device)

    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        # Size against what is actually IN USE, not against the driver's `free`. PyTorch's caching
        # allocator keeps freed blocks in its own pool, so mem_get_info undercounts what is available
        # to this process by exactly that pool -- job 697404 measured 6 GB free where ~14 GB was
        # usable, and paid for it with 67 cache evictions and re-scored banks.
        allocated_b = torch.cuda.memory_allocated()
        cache_bytes = max(int(2 * 2**30), int(total_b - allocated_b - cache_reserve_gb * 2**30))
    else:
        free_b = total_b = allocated_b = 0
        cache_bytes = 2 * 2**30

    distance = CachedSquaredGlobalIEM(
        None, gammas, num_eps=NUM_EPS, seed=DIST_SEED, score_fn=score_fn,
        cache_bytes=cache_bytes)

    notes = [
        f"device {device}  d={LATENT_DIM}  prompt={PROMPT!r} guidance={GUIDANCE} img={IMG}",
        f"N_GAMMA={N_GAMMA} NUM_EPS={NUM_EPS}  denoiser batch capped at {MAX_DENOISER_ROWS} rows",
        f"data scale S={S:.3f} (sweep 695271: 1.015)  gamma=[{gamma_lo:.4g}, {gamma_hi:.4g}] "
        f"x {N_GAMMA} nodes",
        f"probes: {PROBE_SIZE} points from seed {SEED + 9973} (both arms rank THESE points)",
        f"cost per point: {GEN_ROWS} gen + {BANK_ROWS} bank rows = "
        f"{(GEN_ROWS + BANK_ROWS) * S_ROW:.1f} s   bank VRAM {BANK_BYTES_PER_POINT / 2**20:.0f} MB/pt",
        f"VRAM {allocated_b / 2**30:.1f} GB allocated ({(total_b - free_b) / 2**30:.1f} GB reserved) "
        f"of {total_b / 2**30:.1f} GB; bank cache budget {cache_bytes / 2**30:.1f} GB",
    ]
    if verbose:
        for line in notes:
            print(line)
    return Setup(device=device, p_gen=p_gen, distance=distance, gammas=gammas,
                 data_scale=S, notes=notes)


# =====================================================================================================
# Results IO
# =====================================================================================================


def results_path(out_dir: str, arm: str) -> str:
    return os.path.join(out_dir, f"auto_r_results_{arm}.json")


def save_results(out_dir: str, arm: str, payload: dict) -> str:
    """Atomic write, so a timeout mid-save cannot leave a truncated file behind."""
    import json

    path = results_path(out_dir, arm)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    print(f"  saved -> {os.path.basename(path)}")
    return path


def config_dict() -> dict:
    """The pinned configuration, recorded in every results file so two arms can be checked for drift."""
    return {"model": MODEL_ID, "prompt": PROMPT, "guidance": GUIDANCE, "img": IMG,
            "d": LATENT_DIM, "n_gamma": N_GAMMA, "num_eps": NUM_EPS, "n_steps": N_STEPS,
            "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
            "max_denoiser_rows": MAX_DENOISER_ROWS, "dist_seed": DIST_SEED, "seed": SEED,
            "probe_size": PROBE_SIZE, "tau_target": TAU_TARGET, "s_row": S_ROW}
