# Global SUVR Window Explorer

This folder contains a standalone interactive website derived from `analysis/branch_global_suvr_single_visit.ipynb`.

What it does:

- Computes `global_suvr` as the row-wise mean of all cortical `CTX_*` SUVR columns.
- Draws the notebook-style branch trajectory background from cluster centers.
- Lets you move a `global_suvr` window, including dragging the bar between the two handles.
- Auto-loads every `analysis/trial2_*_pre.csv` dataset into the selector.

Usage:

1. Regenerate the embedded data bundle after CSV changes:

```bash
python website/branch_global_suvr/export_data.py
```

2. Open `website/branch_global_suvr/index.html` directly in a browser.

Optional local server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/website/branch_global_suvr/`.
