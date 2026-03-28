# Contributing — March Mania ML

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

For notebooks and full training stack (PyTorch, XGBoost, Jupyter):

```bash
pip install -r requirements.txt
```

Set `MM_DATA_DIR` if Kaggle competition CSVs are not in `./data`. See `src/paths.py` and the [README](README.md).

## Tests

```bash
python -m pytest tests -q
```

Run tests before opening a PR that changes `src/` or `tests/`.

## Code and style

- Match existing module layout and naming in `src/`.
- Prefer type hints and focused functions; keep public APIs consistent with `submission.py` / `inference.py` if you touch feature or model paths.
- Do not commit large competition CSVs; use `.gitignore` and document data location in the PR if needed.

## Notebooks

EDA and experiments live under `src/*.ipynb`. Clear outputs before commit if your team policy requires clean notebooks, or document that notebooks are meant to retain outputs.

## Artifacts and downstream

Trained files under `src/models/` are consumed by **march-mania-backend** when copied there. If you change artifact names or formats, update the README “Outputs consumed by other repos” section and any backend loaders in the same change set (or note the breaking change in the PR).

## Pull requests

Keep the scope small, describe motivation and any evaluation or metric impact, and confirm `pytest` passes locally.

For contributions that span the web app or API, see the monorepo [CONTRIBUTING.md](../CONTRIBUTING.md) at the repository root.
