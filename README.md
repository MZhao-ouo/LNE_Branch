# LNE Branch

LNE Branch is a branch-aware extension of Longitudinal Neighbourhood Embedding for learning and analyzing progression trajectories from longitudinal regional amyloid-PET SUVR features.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Set the dataset and training options in `config.yaml`. Cohort CSV files, derived splits, checkpoints, and generated results are intentionally not included.

The repository contains the core model and training scripts, preprocessing notebooks under `data/`, analysis notebooks under `analysis/`, and a data-free interactive explorer under `website/`.

## Upstream LNE

This project builds on:

- J. Ouyang et al., “Self-Supervised Longitudinal Neighbourhood Embedding,” *MICCAI 2021*, pp. 80–89. [doi:10.1007/978-3-030-87196-3_8](https://doi.org/10.1007/978-3-030-87196-3_8)
- Original implementation: [ouyangjiahong/longitudinal-neighbourhood-embedding](https://github.com/ouyangjiahong/longitudinal-neighbourhood-embedding)
