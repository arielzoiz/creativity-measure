"""Static check: every symbol a notebook imports must actually exist. Seconds, on the login node.

WHY THIS EXISTS. Job 782111 burned 28.8 min of A6000 -- 26.4 of it rebuilding the reference bank --
and then died on `ImportError: cannot import name 'posterior_z_ddpm_step'`. The Phase 6 cell had that
import at MODULE level while its body sat behind `if DO_STRENGTH:`, so reverting the library broke a
cell that no longer even ran. Nothing about that needed a GPU to discover.

It is the third of four failures in this notebook's history that was statically catchable:
  781449  bare `auth_check` in the slurm preflight     -> stale HF token, 401
  781469  bare `auth_check` in notebook cell 1         -> OfflineModeIsEnabled (the fix for the first)
  782111  import of a symbol deleted from the library  -> ImportError, after 28.8 min

The reference bank rebuilds EVERY job (~26 min, it is not cached), so every avoidable failure costs
that much before it even reaches the new code. Hence: run this before `sbatch`, and the .slurm runs it
too, before staging and before the GPU does anything expensive.

WHAT IT CHECKS. Every `import X` / `from X import a, b` at any depth in every code cell, against the
real modules. Deliberately NOT an execution: it imports the target modules (cheap, CPU-only) and looks
up the attributes, so it cannot be fooled by a name that only exists behind a runtime branch -- which
is exactly the failure mode above.

Usage:  python check_notebook_imports.py <notebook.ipynb> [more.ipynb ...]
Exit 0 if every symbol resolves, 1 otherwise.
"""

import ast
import importlib
import json
import sys


def check(path: str) -> list[str]:
    nb = json.load(open(path))
    problems: list[str] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            problems.append(f"cell {idx}: SyntaxError: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except Exception as exc:                       # noqa: BLE001 - report, never raise
                        problems.append(f"cell {idx} L{node.lineno}: import {alias.name} -> "
                                        f"{type(exc).__name__}: {exc}")
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:                  # relative import, not ours to resolve
                    continue
                try:
                    mod = importlib.import_module(node.module)
                except Exception as exc:                           # noqa: BLE001
                    problems.append(f"cell {idx} L{node.lineno}: from {node.module} -> "
                                    f"{type(exc).__name__}: {exc}")
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if not hasattr(mod, alias.name):
                        problems.append(f"cell {idx} L{node.lineno}: "
                                        f"`from {node.module} import {alias.name}` -- NOT FOUND")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for path in sys.argv[1:]:
        problems = check(path)
        if problems:
            bad = 1
            print(f"FAIL {path}")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"ok   {path}: every imported symbol resolves")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
