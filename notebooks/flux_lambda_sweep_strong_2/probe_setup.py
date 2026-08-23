"""Localize where run 2.1's setup stalls, and leave behind the reference latents.

WHY. Job 754633 held an A6000 for the full 6 h and produced NOTHING on disk: no output notebook
(nbconvert writes it only after the last cell), no refs_R64_seed7.pt, no partial checkpoint. The .out
log ends at "=== executing ... ===". So the run stalled somewhere in cells 1-4 and left no evidence
of where. That is the defect this probe attacks first -- a second blind 6 h submission would most
likely reproduce the same silence.

HOW. The notebook's cells 1-4 are EXEC'd from the .ipynb itself, not copied here: the reference
latents must come from bit-identical code (same G, MAX_DENOISER_ROWS=24, seed 7, sigma schedule,
prompt, guidance) or the resume guard in cell 5 will reject them. Copying the code would invite
exactly the drift the guard exists to catch.

Two things make a stall visible where nbconvert made it invisible:
  * every phase boundary prints with flush=True to stdout, which slurm writes straight to the .out;
  * faulthandler.dump_traceback_later(300, repeat=True) dumps the Python stack to stderr every 5 min,
    so a hang names its own frame -- model load, an HF network call, a CUDA sync, whatever it is.

ON SUCCESS this is not throwaway work: cell 4 writes refs_R64_seed7.pt, which every later leg loads
instead of regenerating, taking G out of the reproducibility chain for good.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import time

NB = "flux_dev_strong_tilt_sweep_2_1.ipynb"
CELLS = (1, 2, 3, 4)          # setup, pipeline, denoiser/G/score, knobs + refs + reward bank
DUMP_EVERY_S = 300

T0 = time.time()


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] +{time.time() - T0:7.1f}s  {msg}", flush=True)


# Stack dump every 5 min, to stderr, without killing the process. This is the whole point of the
# probe: if it hangs, the traceback says where instead of leaving a 6 h hole.
faulthandler.enable()
faulthandler.dump_traceback_later(DUMP_EVERY_S, repeat=True, exit=False)

stamp(f"probe start   cwd={os.getcwd()}")
stamp(f"python {sys.version.split()[0]}   notebook {NB}")

with open(NB) as fh:
    nb = json.load(fh)

# One shared namespace, so cell N sees everything cells 1..N-1 defined -- same as a kernel.
ns: dict[str, object] = {"__name__": "__main__"}

for idx in CELLS:
    cell = nb["cells"][idx]
    assert cell["cell_type"] == "code", f"cell {idx} is {cell['cell_type']}, expected code"
    src = "".join(cell["source"])
    stamp(f"=== cell {idx} START ({len(src.splitlines())} lines) ===")
    t_cell = time.time()
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), ns)
    except BaseException as exc:
        stamp(f"=== cell {idx} RAISED {type(exc).__name__}: {exc} ===")
        raise
    stamp(f"=== cell {idx} DONE in {time.time() - t_cell:.1f}s ===")

faulthandler.cancel_dump_traceback_later()

refs = os.path.join(str(ns.get("SWEEP_DIR", ".")), f"refs_R{ns.get('R')}_seed7.pt")
stamp(f"ALL SETUP CELLS COMPLETED in {(time.time() - T0) / 60:.1f} min")
stamp(f"refs on disk: {os.path.exists(refs)}  -> {refs}")
if os.path.exists(refs):
    stamp(f"refs size {os.path.getsize(refs) / 2**20:.1f} MB "
          f"(later legs load these instead of running G)")
