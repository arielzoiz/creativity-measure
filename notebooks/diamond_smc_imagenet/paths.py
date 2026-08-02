"""Asset and environment resolution for the Diamond Maps SMC run.

Run-specific plumbing, deliberately kept out of the library: where the checkpoints and the upstream
clone live is a property of *this machine*, not of `creativity_measure`.

The checkpoints are **not copied into the project**. They stay in the HuggingFace cache
(`$HF_HOME`, redirected off the 2 GB home in `~/.cshrc`) and are addressed by path — upstream's
`repo_paths.ckpt_path` default is overridden at config time by `DiamondMapsBackend`. That keeps the
diamond_maps clone a pristine checkout and the 10.9 GB out of the project tree.
"""

from __future__ import annotations

import os

REPO_ID = "MonkeyDoug/diamond-maps"
BASE_CKPT_FILE = "ckpt/sit_assets/SiT-XL-2.pkl"           # base: transitions + IEM score model
POSTERIOR_CKPT_FILE = "ckpt/ImageNet-DiamondMap-B2.pkl"   # posterior: one-NFE lookahead

DEFAULT_DIAMOND_MAPS_ROOT = "/home/dcor/arielzoizner/repos/diamond_maps"


def diamond_maps_root() -> str:
    """The upstream clone. Imported live from `<root>/posterior_diamond_maps/py`; never copied."""
    root = os.environ.get("DIAMOND_MAPS_ROOT", DEFAULT_DIAMOND_MAPS_ROOT)
    if not os.path.isdir(os.path.join(root, "posterior_diamond_maps", "py")):
        raise FileNotFoundError(
            f"{root} is not a diamond_maps clone. Clone it or set $DIAMOND_MAPS_ROOT:\n"
            f"  git clone https://github.com/PeterHolderrieth/diamond_maps {root}"
        )
    return root


def checkpoint_paths(local_files_only: bool = True) -> tuple[str, str]:
    """``(base_ckpt, posterior_ckpt)`` resolved inside the HF cache.

    With ``local_files_only=True`` (the default, and what a compute node should use) this only
    resolves what is already cached and never touches the network. Download once, from a login node::

        hf download MonkeyDoug/diamond-maps \\
            --include 'ckpt/sit_assets/SiT-XL-2.pkl' 'ckpt/ImageNet-DiamondMap-B2.pkl'
    """
    from huggingface_hub import hf_hub_download

    return tuple(                                             # type: ignore[return-value]
        hf_hub_download(REPO_ID, f, local_files_only=local_files_only)
        for f in (BASE_CKPT_FILE, POSTERIOR_CKPT_FILE)
    )


def preflight(min_gb: float = 40.0, require_bf16: bool = True) -> dict:
    """Fail in seconds rather than after minutes of model loading.

    Checks the things that have actually broken runs here before: no CUDA device, a card too small
    for the XL/2 transformer, a card without native bf16 (the configs use `compute_dtype=bfloat16`,
    and the login node's TITAN Xp is Pascal), and missing checkpoints.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible — this must run on a GPU node via Slurm")
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.mem_get_info()[1] / 2**30
    if gb < min_gb:
        raise RuntimeError(f"need a >={min_gb:.0f}GB card for SiT-XL/2, got {name} with {gb:.0f} GB")
    if require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} lacks native bf16; the upstream configs compute in bfloat16")

    base, posterior = checkpoint_paths()
    for p in (base, posterior):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    import jax

    return {
        "gpu": name,
        "gpu_gb": round(gb, 1),
        "jax_devices": [str(d) for d in jax.devices()],
        "base_ckpt": base,
        "posterior_ckpt": posterior,
        "diamond_maps_root": diamond_maps_root(),
    }
