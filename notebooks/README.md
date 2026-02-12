# Notebooks

- **03_label_flipping_robustness.ipynb** – Label-flipping attack robustness (trust-based filtering).
- **04_gradient_leakage.ipynb** – Gradient leakage attack resilience (uses `src` data loaders and models).
- **05_backdoor_resilience.ipynb** – Backdoor attack resilience (MIC analysis and noise allocation).

## Running

Open and run in Jupyter (Lab or Notebook) with the kernel’s working directory set to this `notebooks/` folder so `../src` resolves to the project’s `src/`.

## Testing all notebooks

From the **repo root** (the `LDP-MIC` folder that contains `src/` and `notebooks/`):

```bash
python notebooks/run_notebooks_test.py
```

Requirements: `jupyter`, `nbconvert`, and the rest of the project (PyTorch, etc.). If you see a **PyTorch DLL error** (e.g. `c10.dll`) on Windows, fix the PyTorch environment (e.g. install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) or use a conda env with a working `torch` install), then run the script again.
