"""Bridge to the authors' JAX implementation of Diamond Maps (arXiv:2602.05993).

Supplies the two generative operations :mod:`creativity_measure.diamond_smc` needs, plus the marginal
``score_fn`` its IEM reward needs, by calling the upstream ``posterior_diamond_maps`` code **live** —
nothing is copied, vendored or edited. Upstream stays a pristine git checkout.

Why a bridge and not a port
---------------------------
Porting the checkpoints to PyTorch would mean reimplementing not just SiT but the whole GLASS /
diamond sampling stack (``InnerInterpolant``, ``calc_xbar_s0``, ``calc_s``, ``calc_x_t_prime``, the
``s``-schedule, ``reverse_time``, the CFG token) — thousands of lines of subtle JAX with no reference
outputs to check against. It is unnecessary here: the IEM reward is **gradient-free by design**, so
only arrays ever cross the boundary, never autodiff. The authors themselves ship torch alongside jax.

Two networks, three jobs
------------------------
======================  ==========================  ==================================================
slot                    checkpoint                  used for
======================  ==========================  ==================================================
base (``main``)         ``SiT-XL-2.pkl``            Algorithm 2 line 6 (DDPM transition, via GLASS)
                                                    **and** the marginal score for the IEM reward
posterior (``sup``)     ``ImageNet-DiamondMap-B2``  Algorithm 2 line 9 (one-NFE lookahead)
======================  ==========================  ==================================================

The base doing double duty is deliberate: the score model *is* the network generating the particles,
with the same ``label`` and ``cfg_scale``, so the reward cannot drift away from the distribution being
sampled.

The score conversion
--------------------
The upstream interpolant is ``linear``: ``x_t = (1-t)·x_0 + t·x_1`` with ``x_0`` noise and ``x_1``
data, so ``t = 1`` is data. This repo's gamma convention is ``Y = gamma·x + sqrt(gamma)·W``, i.e.
``sigma_EDM = 1/sqrt(gamma)``. Matching the two::

    sigma_EDM(t) = (1 - t) / t        gamma(t) = (t / (1 - t))^2        t(gamma) = sqrt(gamma) / (1 + sqrt(gamma))

and the denoiser is ``E[x_1 | x_t] = x_t + (1-t)·v_t(x_t)``, which upstream already implements as
``flow.GlassFlow._denoiser``. So this module only has to invert ``sigma -> t``, rescale, and hand the
result to :func:`creativity_measure.distances.edm_adapter.edm_score_fn`; there is no bespoke
velocity-to-score arithmetic anywhere.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.dlpack   # submodule: not imported by `import torch` alone
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.distances.edm_adapter import edm_score_fn

_MISSING_DEPS_MSG = (
    "DiamondMapsBackend needs the upstream Diamond Maps stack (jax, flax, optax, ml_collections, "
    "tensorflow, tensorflow_datasets, diffusers) and a checkout of "
    "https://github.com/PeterHolderrieth/diamond_maps. Point `repo_root` (or $DIAMOND_MAPS_ROOT) at "
    "the clone and install the deps from posterior_diamond_maps/requirements.txt."
)

# The image/latent grid these ImageNet configs operate on.
LATENT_SHAPE: tuple[int, int, int] = (4, 32, 32)
# configs.imagenet.smc variant index: 0 = flow posterior (their baseline), 1 = diamond map posterior.
DIAMOND_POSTERIOR_VARIANT: int = 1


def _add_repo_to_path(repo_root: str | None) -> str:
    """Put ``<repo_root>/posterior_diamond_maps/py`` on ``sys.path`` and return the repo root.

    Upstream imports ``common.*`` and ``configs.*`` as top-level packages ("run commands from the
    repo root"), so its ``py/`` directory has to be importable. Their ``repo_paths`` then resolves
    ``ckpt/`` relative to its own file, which we override anyway — checkpoints live in the HF cache.
    """
    root = repo_root or os.environ.get("DIAMOND_MAPS_ROOT")
    if not root:
        raise ValueError(
            "repo_root not given and $DIAMOND_MAPS_ROOT not set; both must point at a clone of "
            "https://github.com/PeterHolderrieth/diamond_maps"
        )
    py_dir = os.path.join(root, "posterior_diamond_maps", "py")
    if not os.path.isdir(py_dir):
        raise FileNotFoundError(f"{py_dir} does not exist — is {root} really the diamond_maps clone?")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    return root


def _as_variables(tree: Any) -> Any:
    """Normalize a checkpoint params tree into a Flax variables dict.

    ``load_params_from_checkpoint`` returns whatever the training state stored, which may already be
    a full variables dict (``{"params": ...}``) or a bare params tree. ``Module.apply`` needs the
    former, so accept both rather than guessing.
    """
    if isinstance(tree, dict) and "params" in tree:
        return tree
    return {"params": tree}


class DiamondMapsBackend:
    """Drives the upstream GLASS base sampler and diamond-map posterior from PyTorch.

    Satisfies :class:`creativity_measure.diamond_smc.DiamondMapBackend`.

    ``label`` and ``cfg_scale`` are **required and have no defaults**: they define the prior ``p``
    that the whole experiment is about, so the library must never pick one silently. Choose them at
    the call site (i.e. in the notebook).

    Args:
        label:      ImageNet class index conditioning the model. ``0..999`` selects a class (e.g. 207
                    = golden retriever); ``1000`` is the null / CFG token, giving the unconditional
                    marginal. There is no umbrella "dog" class — ImageNet-1k has ~120 dog breeds.
        cfg_scale:  classifier-free guidance scale for the **base** network. ``1.0`` samples the true
                    ``p(.|label)``; the upstream default ``4.0`` sharpens it into a different
                    distribution, which also distorts the IEM geometry the reward is built from.
        base_ckpt:  path to ``SiT-XL-2.pkl``.
        posterior_ckpt: path to ``ImageNet-DiamondMap-B2.pkl``.
        repo_root:  the diamond_maps clone; defaults to ``$DIAMOND_MAPS_ROOT``.
        n_steps:    number of DDPM transitions ``N``; fixes the time grid ``linspace(0, 1, N+1)``.
        base_inner_steps: GLASS inner-flow steps per transition (upstream's ``--base_inner_steps``).
        mc_inner_steps: inner steps for the posterior lookahead; 1 is the one-NFE diamond map.
        device/dtype: the torch side the arrays are handed back on.
        seed:       seeds this backend's JAX key stream.

    Note:
        The backend is **stateful in its RNG**: each call splits the instance's JAX key. Runs are
        reproducible when the call sequence is (as in the sampler loop), but the methods are not pure.
    """

    def __init__(
        self,
        *,
        label: int,
        cfg_scale: float,
        base_ckpt: str,
        posterior_ckpt: str,
        repo_root: str | None = None,
        n_steps: int = 6,
        base_inner_steps: int = 8,
        mc_inner_steps: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.repo_root = _add_repo_to_path(repo_root)
        try:
            import jax
            import jax.numpy as jnp
            from common import latent_utils, sampling, state_utils
            from common import flow as flow_mod
            from common import interpolant as interp_mod
            from configs.imagenet import smc as smc_cfg
        except ImportError as e:                              # pragma: no cover - needs the extra stack
            raise ImportError(_MISSING_DEPS_MSG) from e

        self._jax = jax
        self._jnp = jnp
        self._sampling = sampling
        self._latent_utils = latent_utils
        self._flow_mod = flow_mod

        self.label = int(label)
        self.cfg_scale = float(cfg_scale)
        self.n_steps = int(n_steps)
        self.device = torch.device(device)
        self.dtype = dtype
        self._dlpack_ok: bool | None = None

        for path, name in ((base_ckpt, "base_ckpt"), (posterior_ckpt, "posterior_ckpt")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} not found: {path}")

        # --- config, with p chosen by the caller ---------------------------------------------------
        cfg = smc_cfg.get_config(DIAMOND_POSTERIOR_VARIANT, os.getcwd())
        with cfg.unlocked():
            cfg.network.sit_cfg_scale = self.cfg_scale
            cfg.network.load_path = base_ckpt
            cfg.sup_network.load_path = posterior_ckpt
        self.cfg = cfg

        # --- networks, WITHOUT setup_state ---------------------------------------------------------
        # `launchers.learn.setup_state` builds a TFDS pipeline purely to get an example input for
        # shape-initializing randomly initialized networks, and eagerly loads Inception for FID.
        # Neither applies here: both of our networks come from checkpoints, so `ex_input` is never
        # touched (state_utils `_initialize_main_model` returns early on use_glass, and
        # `_load_supervisor_model` never had an ex_input). Skipping it is what removes the dependency
        # on the gated ImageNet latent dataset.
        self._net = state_utils._build_top_level_model(cfg.network, cfg.problem.interp_type)
        self._sup_net, sup_params = state_utils._load_supervisor_model(cfg)
        base_params = state_utils.load_params_from_checkpoint(
            cfg.network.load_path, ema_factor=cfg.logging.ema_factor
        )
        self._base_vars = _as_variables(base_params)
        self._sup_vars = _as_variables(sup_params)

        statics = state_utils.StaticArgs(
            net=self._net,
            schedule=None,
            loss=None,
            get_loss_fn_args=None,
            train_step=None,
            ds=None,
            interp=interp_mod.setup_interpolant(cfg.problem.interp_type),
            sample_rho0=None,
            inception_fn=None,
            # decode_fn stays None here on purpose: see `_decode_fn`. Building it pulls in the SD-VAE
            # via diffusers, which the SMC never needs — pixels are entered exactly once, at the end.
            decode_fn=None,
            sup_net=self._sup_net,
            sup_params=self._sup_vars,
        )
        self._statics = statics
        self._decode_fn_cached: Any = None

        # --- the two upstream callables ------------------------------------------------------------
        self.ts = jnp.linspace(0.0, 1.0, self.n_steps + 1)
        self._base_spec = sampling.build_sampler_spec(
            cfg,
            statics,
            sample_type=sampling.SampleType.GLASS,
            outer_steps=self.n_steps,
            inner_steps=int(base_inner_steps),
            ts_override=self.ts,
            network_slot="main",
        )
        self._sample_x1 = sampling.make_sample_x1_fn(
            cfg,
            statics,
            inner_steps=int(mc_inner_steps),
            posterior_sample_type=sampling.SampleType.DIAMOND_EARLY_STOP,
            network_slot="sup",
        )
        # Upstream's whole model stack is written for a SINGLE sample (see
        # network_utils.SiTFlow.process_inputs, which adds the leading axis itself); batching is
        # always vmap on the outside. Two nested vmaps: over particles, then over the K noise keys.
        self._batch_sample_x1 = jax.jit(
            jax.vmap(
                jax.vmap(self._sample_x1, in_axes=(None, None, None, None, 0)),
                in_axes=(None, 0, None, 0, 0),
            )
        )
        self._batch_denoise = jax.jit(
            jax.vmap(
                lambda variables, t, x, lab: self._net.apply(
                    variables, t, x, label=lab, method=flow_mod.GlassFlow._denoiser
                ),
                in_axes=(None, None, 0, 0),
            )
        )
        self._step_fns: dict[int, Callable] = {}

        # Two INDEPENDENT key streams, one per model call. Sharing one stream would make the base
        # trajectory depend on how many lookahead draws were taken, so a lambda=0 run (where the
        # lookahead is computed but multiplied by zero) would silently diverge from a pure base run
        # and the reproduction check would be untestable. Splitting them makes lambda=0 exact.
        self._seed = int(seed)
        self._key_base = jax.random.PRNGKey(self._seed)
        self._key_post = jax.random.PRNGKey(self._seed + 1)
        self._torch_gen = torch.Generator(device=self.device)
        self._torch_gen.manual_seed(self._seed)

    # ------------------------------------------------------------------------------------------
    # Array bridge
    # ------------------------------------------------------------------------------------------

    def _to_torch(self, x: Any) -> Tensor:
        """JAX array -> torch tensor, zero-copy via dlpack when available.

        Falls back to a host round-trip through numpy, which is bit-exact for float32 and costs
        microseconds at these sizes (a full lookahead batch is ~1 MB). Correctness first: the
        fallback is used permanently once dlpack has failed even once.
        """
        x = self._jnp.asarray(x, dtype=self._jnp.float32)
        if self._dlpack_ok is not False:
            try:
                out = torch.utils.dlpack.from_dlpack(x)
                self._dlpack_ok = True
                return out.to(device=self.device, dtype=self.dtype)
            except Exception:                                   # pragma: no cover - platform dependent
                self._dlpack_ok = False
        import numpy as np

        return torch.from_numpy(np.asarray(x)).to(device=self.device, dtype=self.dtype)

    def _to_jax(self, x: Tensor) -> Any:
        """torch tensor -> JAX array, mirroring :meth:`_to_torch`."""
        x = x.detach().to(torch.float32).contiguous()
        if self._dlpack_ok is not False:
            try:
                return self._jnp.asarray(self._jax.dlpack.from_dlpack(x))
            except Exception:                                   # pragma: no cover - platform dependent
                self._dlpack_ok = False
        return self._jnp.asarray(x.cpu().numpy())

    def _next_keys(self, stream: str, *shape: int) -> Any:
        """Split off a fresh block of PRNG keys, advancing one of the two independent streams.

        ``stream`` is ``"base"`` or ``"post"``; see the constructor for why they are separate.
        """
        attr = "_key_base" if stream == "base" else "_key_post"
        key, sub = self._jax.random.split(getattr(self, attr))
        setattr(self, attr, key)
        n = 1
        for s in shape:
            n *= s
        keys = self._jax.random.split(sub, n)
        return keys.reshape(*shape, -1)

    # ------------------------------------------------------------------------------------------
    # DiamondMapBackend protocol
    # ------------------------------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return LATENT_SHAPE

    def reset_rng(self, seed: int) -> None:
        """Rewind both RNG streams, so the next run repeats the previous one's randomness.

        Needed for a **controlled lambda sweep**: this backend is stateful in its RNG, so reusing one
        instance across lambdas would give each run different initial particles and different base
        transitions, confounding the tilt's effect with the noise draw. Call this before each run to
        vary only lambda.
        """
        self._seed = int(seed)
        self._key_base = self._jax.random.PRNGKey(self._seed)
        self._key_post = self._jax.random.PRNGKey(self._seed + 1)
        self._torch_gen.manual_seed(self._seed)

    def init_particles(self, n: int) -> Float[Tensor, "n 4 32 32"]:
        """``x_0 ~ N(0, I)`` — Algorithm 2 line 2. Drawn in torch, off this backend's generator."""
        return torch.randn(
            (n, *LATENT_SHAPE), generator=self._torch_gen, device=self.device, dtype=self.dtype
        )

    def _labels(self, n: int) -> Any:
        return self._jnp.full((n,), self.label, dtype=self._jnp.int32)

    def _step_fn(self, step_idx: int) -> Callable:
        """A jitted single outer step of the base sampler, specialized to ``step_idx``.

        ``step_idx`` selects ``spec.ts[step_idx]`` and so must be static; there are only ``n_steps``
        of them, so we cache one compiled function per step rather than tracing the index.
        """
        if step_idx not in self._step_fns:
            jax, sampling, spec = self._jax, self._sampling, self._base_spec

            def run(variables, batch_x_t, batch_keys, batch_label):
                result, _ = sampling.run_sampler_step(
                    spec=spec,
                    variables=variables,
                    step_idx=step_idx,
                    batch_x_t=batch_x_t,
                    batch_prng_key=batch_keys,
                    batch_label=batch_label,
                    batch_measurement=None,
                    return_traj=False,
                )
                return result.batch_x_final

            self._step_fns[step_idx] = jax.jit(run)
        return self._step_fns[step_idx]

    def base_step(
        self, x_t: Float[Tensor, "n 4 32 32"], step_idx: int
    ) -> Float[Tensor, "n 4 32 32"]:
        """Algorithm 2 line 6 — one DDPM transition, realized by the GLASS inner flow.

        With the spec's default resampling (identity) and guidance (zero), ``run_sampler_step``
        reduces to exactly ``calc_xbar_s0 -> inner GLASS integration -> calc_x_t_prime``, so
        ``batch_x_final == batch_x_t_prime``. We drive the loop; upstream's step is untouched.
        """
        n = x_t.shape[0]
        out = self._step_fn(int(step_idx))(
            self._base_vars, self._to_jax(x_t), self._next_keys("base", n), self._labels(n)
        )
        return self._to_torch(out)

    def posterior_sample(
        self, x_t: Float[Tensor, "n 4 32 32"], step_idx: int, mc_samples: int
    ) -> Float[Tensor, "nk 4 32 32"]:
        """Algorithm 2 line 9 — ``mc_samples`` one-NFE draws of ``x_1`` per particle.

        Returns ``repeat_interleave`` order: ``.view(n, mc_samples, ...)`` groups by particle, which
        is what :func:`~creativity_measure.diamond_smc._soft_value` assumes.
        """
        n = x_t.shape[0]
        t_prime = self.ts[int(step_idx) + 1]
        z = self._batch_sample_x1(
            self._sup_vars,
            self._to_jax(x_t),
            t_prime,
            self._labels(n),
            self._next_keys("post", n, int(mc_samples)),
        )                                                        # (n, K, 4, 32, 32)
        return self._to_torch(z).reshape(n * int(mc_samples), *LATENT_SHAPE)

    # ------------------------------------------------------------------------------------------
    # Score model and helpers
    # ------------------------------------------------------------------------------------------

    def denoiser(
        self, y_sigma: Float[Tensor, "B 4 32 32"], sigma: Float[Tensor, "B"]
    ) -> Float[Tensor, "B 4 32 32"]:
        """``E[X | y_sigma]`` in the EDM convention, from the base network.

        ``edm_score_fn`` always calls this with a single scalar ``sigma`` broadcast over the batch
        (it derives it from one scalar ``gamma``), so ``t`` is scalar and the batch shares it.
        """
        sigma_scalar = float(sigma.reshape(-1)[0])
        t = 1.0 / (1.0 + sigma_scalar)                 # invert sigma = (1 - t) / t
        x_t = self._to_jax(y_sigma) * t                # y_sigma = x_t / t
        x_pred = self._batch_denoise(
            self._base_vars, self._jnp.asarray(t, dtype=self._jnp.float32),
            x_t, self._labels(y_sigma.shape[0]),
        )
        return self._to_torch(x_pred)

    @property
    def score_fn(self) -> ScoreFn:
        """The marginal score ``grad_y log p_Y(y, gamma)`` as a plain torch callable.

        Handed straight to ``SquaredGlobalIEMDistance(score_fn=...)``, which never learns that JAX
        exists. All the schedule arithmetic lives in ``edm_adapter``.
        """
        return edm_score_fn(self.denoiser, img_shape=LATENT_SHAPE)

    def gamma_of_t(self, t: float) -> float:
        """``gamma(t) = (t / (1 - t))^2`` — the reward's integration variable at model time ``t``."""
        return (t / (1.0 - t)) ** 2

    def gamma_window(self, t_min: float = 0.01, t_max: float = 0.99) -> tuple[float, float]:
        """Valid ``gamma`` range, from the model times the denoiser is trained on.

        ``edm_adapter`` requires the caller to keep ``gamma`` inside the denoiser's usable ``sigma``
        range; ``t`` must stay strictly inside ``(0, 1)`` or ``sigma`` blows up / collapses.
        """
        return self.gamma_of_t(t_min), self.gamma_of_t(t_max)

    def sample_refs(self, n: int) -> Float[Tensor, "n d"]:
        """``n`` samples from ``p``, flattened — the reference set for the IEM reward.

        A full base-sampler run (no tilt), so the references are drawn from exactly the distribution
        the particles are drawn from, with no ImageNet download involved.
        """
        x = self.init_particles(n)
        for step in range(self.n_steps):
            x = self.base_step(x, step)
        return x.reshape(n, -1)

    @property
    def _decode_fn(self) -> Any:
        """The SD-VAE decoder, built on first use.

        Deliberately lazy. It is the only thing in this backend that needs `diffusers`, and the whole
        algorithm runs in latent space — so a `diffusers`/torch version clash must not be able to stop
        an SMC run that is never going to look at pixels. Loading it costs ~20 s and ~350 MB.
        """
        if self._decode_fn_cached is None:
            self._decode_fn_cached = self._latent_utils.get_decode_fn(self.cfg)
            if self._decode_fn_cached is None:
                raise RuntimeError("upstream reports this config is not a latent target")
        return self._decode_fn_cached

    def decode(self, latents: Float[Tensor, "n d"]) -> Float[Tensor, "n 3 H W"]:
        """Latents -> pixel images in ``[0, 1]``, via upstream's Flax SD-VAE.

        Upstream already divides by ``cfg.problem.latent_scale`` (0.18215) inside
        ``maybe_decode_latents_chunked``, so do not pre-scale.
        """
        n = latents.shape[0]
        x = self._to_jax(latents.reshape(n, *LATENT_SHAPE))
        img = self._latent_utils.decode_batch_to_nhwc(
            self.cfg, x, self._decode_fn, unnormalize=True
        )                                                        # (n, H, W, 3), [0, 1]
        return self._to_torch(img).permute(0, 3, 1, 2).clamp(0.0, 1.0)
