"""Execute a notebook in a plain Python process -- no Jupyter kernel, no ZMQ, no connection file.

WHY THIS EXISTS. Three jobs (756120, 756173, 756235) died before cell 1 with
"Kernel didn't respond in N seconds" -- at 60 s, at 600 s, and then three times in a row at 600 s
under a retry loop. The kernel process itself always came up (its IPKernelApp banner is in every
log); what never completed was the client<->kernel handshake. Raising the timeout did not help, and
neither did moving JUPYTER_RUNTIME_DIR (where the connection file lives) onto node-local disk. Both
of those were hypotheses about a handshake I could not observe from outside.

Rather than keep guessing at it, this removes the handshake from the pipeline. The approach is not
speculative: probe_setup.py used exactly this exec-the-cells technique on this cluster, on this node,
and ran the notebook's setup cells end to end -- model load, reference generation, score bank --
while papermill could not start a kernel at all.

WHAT IS LOST vs papermill: rich outputs (inline images, display_data, execution timings). Cells that
matter write their artifacts to DISK anyway -- checkpoints as .pt, the grid as .png -- so nothing
scientific depends on the notebook's embedded outputs. Text output is preserved per cell, so the
saved notebook still reads as a record of the run.

WHAT IS KEPT: live streaming to the job log (the thing nbconvert lacked and that cost 754633 six
silent hours), an output notebook written after EVERY cell (so a kill leaves a partial record rather
than nothing), and a faulthandler watchdog.

Usage:  python run_notebook.py <input.ipynb> <output.ipynb>
Exit code is 0 only if every code cell completed.
"""

from __future__ import annotations

import contextlib
import faulthandler
import io
import json
import os
import sys
import time
import traceback
from typing import Any

DUMP_EVERY_S = 1800


class _Tee(io.TextIOBase):
    """Write to the real stdout AND a capture buffer.

    Flushes on every write: the job log is a file, not a tty, so without this the run would look
    silent for minutes at a time -- which is the exact failure mode (754633) this script exists to
    prevent.
    """

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _stamp(msg: str, t0: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] +{(time.time() - t0) / 60:6.1f}m  {msg}",
          file=sys.__stdout__, flush=True)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    nb_in, nb_out = sys.argv[1], sys.argv[2]
    t0 = time.time()

    faulthandler.enable()
    faulthandler.dump_traceback_later(DUMP_EVERY_S, repeat=True, exit=False)

    with open(nb_in) as fh:
        nb = json.load(fh)

    def save() -> None:
        """Atomic write, so a kill mid-save cannot truncate the record."""
        tmp = f"{nb_out}.tmp"
        with open(tmp, "w") as fh:
            json.dump(nb, fh, indent=1)
        os.replace(tmp, nb_out)

    ns: dict[str, Any] = {"__name__": "__main__"}
    count = 0
    failed = False

    _stamp(f"executing {os.path.basename(nb_in)} -> {os.path.basename(nb_out)} "
           f"({sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code cells), no kernel", t0)

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        count += 1
        _stamp(f"=== cell {idx} START ===", t0)
        buf = io.StringIO()
        t_cell = time.time()
        try:
            with contextlib.redirect_stdout(_Tee(sys.__stdout__, buf)):
                exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except BaseException:
            tb = traceback.format_exc()
            print(tb, file=sys.__stderr__, flush=True)
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": buf.getvalue()},
                {"output_type": "error", "ename": type(sys.exc_info()[1]).__name__,
                 "evalue": str(sys.exc_info()[1]), "traceback": tb.splitlines()},
            ]
            cell["execution_count"] = count
            save()
            _stamp(f"=== cell {idx} RAISED after {(time.time() - t_cell) / 60:.1f}m ===", t0)
            failed = True
            break
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": buf.getvalue()}]
        cell["execution_count"] = count
        save()                                  # after EVERY cell, so a kill leaves a record
        _stamp(f"=== cell {idx} DONE in {(time.time() - t_cell) / 60:.1f}m ===", t0)

    faulthandler.cancel_dump_traceback_later()
    _stamp(f"{'FAILED' if failed else 'completed'} after {(time.time() - t0) / 60:.1f} min", t0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
