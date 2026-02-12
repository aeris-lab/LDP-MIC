"""Run all notebooks and report pass/fail.

Run from repo root: python notebooks/run_notebooks_test.py

Requires: PyTorch, jupyter, nbconvert, and project deps. On Windows, if you see
c10.dll or DLL load errors, install Visual C++ Redistributable or use a conda env with torch.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def run_notebook(path):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors import CellExecutionError

    nb_path = os.path.join(NOTEBOOKS_DIR, path)
    if not os.path.isfile(nb_path):
        return False, f"File not found: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    try:
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOKS_DIR}})
        return True, None
    except CellExecutionError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    notebooks = [
        "03_label_flipping_robustness.ipynb",
        "04_gradient_leakage.ipynb",
        "05_backdoor_resilience.ipynb",
    ]
    results = {}
    for name in notebooks:
        ok, err = run_notebook(name)
        results[name] = "PASS" if ok else f"FAIL: {err}"
        print(f"{name}: {results[name]}")
    failed = [n for n, r in results.items() if not r.startswith("PASS")]
    if failed and any("c10.dll" in str(r) or "DLL" in str(r) for r in results.values()):
        print("\nPyTorch DLL load error: fix your torch install (e.g. VC++ Redistributable on Windows).")
    sys.exit(1 if failed else 0)
