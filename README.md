# Projet MVA DLMI - Histopathology OOD Classification

Ce README explique rapidement a quoi sert chaque notebook du depot pour que ce soit simple a reprendre.

This project uses `uv` for lightning-fast, reproducible dependency management.

## 1. Environment Setup

Make sure you have `uv` installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
From the project root, run:

```bash
# Install dependencies and create the virtual environment
uv sync

# Activate the environment
source .venv/bin/activate

#Add the kernel in case it doesn't show up automatically
uv run python -m ipykernel install --user --name "mva-spine" --display-name "Spine-Project (.venv)"

#If you ever add a new library (e.g., pandas). This automatically updates pyproject.toml and your environment
uv add pandas 
```

## Data

- Data in `.h5` format is not available on GitHub.
- You must therefore download these data files separately in order to run the full pipeline.

## Project Notebooks

### `getting_started.ipynb`

- Notebook provided by the instructors.
- I haven’t modified it.
- I only ran it once to verify that everything works.

### `MVA_DLMI_adapters.ipynb`

- Corrected lab (reference material).
- Mainly serves as a source of inspiration for the method, structure, and ideas.

### `TP-validation_teacher-version.ipynb`

- Another corrected notebook / teacher version.
- Useful as a basis for comparison or to verify technical choices.

### `explore_data.ipynb`

- Notebook I created to familiarize myself with the data.
- Objective: explore the dataset, examine examples, and verify preprocessing steps.
- It helps understand “what is happening” before coding the main pipeline.
- Detailed contents:
  - configuration/imports and verification that the expected files are present;
  - inspection of the HDF5 structure (how patches and labels are stored);
  - creation of a summary table per patch to facilitate analysis;
  - overall quantitative summary (counts, distributions);
  - class balance and distribution by center plots;
  - pixel analysis on a subsample:
    - intensity histograms by channel,
    - correlations between channels,
    - effect of z-score normalization per patch,
    - identification of the brightest/darkest patches;
  - comparison of the average channel profile between validation and training centers;
  - qualitative visualizations:
    - one image per pair (center, label) for training/validation,
    - random grid on the test domain.
- In practice, this notebook is used to:
  - verify that there are no obvious inconsistencies in the data;
  - guide preprocessing/data augmentation choices before training;
  - document insights regarding domain shift and class balance.

### `my_pipeline.ipynb`

- Main notebook for the project.
- This is where we will work on, test, and refine the solution.
- It is an “experiment runner” centered on DINOv2 + adapters.
- Everything needed for its operation is located in `utils` (modified to support multiple workers).


Translated with DeepL.com (free version)
