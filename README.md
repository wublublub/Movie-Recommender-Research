# Personalized Top-K Movie Recommendation

This repository supports a semester-long machine learning research project on personalized Top-K movie recommendation using the MovieLens dataset. The initial project setup focuses on reproducible data exploration and experimentation; recommendation models will be added in later stages.

## Directory structure

- `data/raw/`: original, immutable MovieLens downloads (not tracked by Git)
- `data/processed/`: cleaned or transformed datasets (not tracked by Git)
- `data/splits/`: reproducible train, validation, and test split definitions
- `notebooks/`: exploratory analysis and research notebooks
- `src/movie_recommender/`: reusable Python package code
- `tests/`: automated tests for package code
- `results/`: generated metrics, figures, and experiment outputs (not tracked by Git)

## Environment setup

Create and activate a Python virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start Jupyter with:

```powershell
jupyter notebook
```
