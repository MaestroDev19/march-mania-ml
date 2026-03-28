# March Mania — ML pipeline

Training, evaluation, and submission generation for **[March Machine Learning Mania 2026](https://kaggle.com/competitions/march-machine-learning-mania-2026)** on Kaggle: predict hypothetical men’s and women’s Division I matchup outcomes from historical NCAA data; scoring uses the **Brier score** on probabilities vs. realized results. This package is **offline research** — it does not serve HTTP traffic. Deployed inference lives in **march-mania-backend** and **march-mania-web**.

**Competition and data context:** In the full monorepo, see [`../details.md`](../details.md) for evaluation rules, submission format (`ID,Pred` with lower `TeamId` first), dataset sections, and field-level file descriptions. The competition and dataset remain subject to [Kaggle’s terms](https://www.kaggle.com/rules).

## Citation

```bibtex
@misc{march-machine-learning-mania-2026,
    author = {Jeff Sonas and Martyna Plomecka and Yao Yan and Addison Howard},
    title = {March Machine Learning Mania 2026},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/march-machine-learning-mania-2026}},
    note = {Kaggle}
}
```

## How to use

1. **Get data** — Download competition CSVs from Kaggle into `./data` (or point elsewhere with `MM_DATA_DIR`).
2. **Environment** — Create a venv and install dependencies (see [Environment](#environment)).
3. **Explore and train** — Open `src/*.ipynb` for EDA and experiments, or import `src` modules from your own scripts. Train models and calibrators; write artifacts under `src/models/` (see [Outputs consumed by other repos](#outputs-consumed-by-other-repos)).
4. **Validate** — Run `python -m pytest tests -q` before you rely on numbers or ship artifacts.
5. **Submissions** — Use the helpers in `src/submission.py` (and notebook workflows) to emit Stage 1 / Stage 2–style CSVs aligned with the competition sample files.
6. **Downstream apps** — Copy trained artifacts into **march-mania-backend**; optional: run `scripts/generate_teams_json.py` (or the `.mjs` variant) to refresh team lists for the web UI, adjusting output paths to match your **march-mania-web** layout if needed.

**Data directory:** Set `MM_DATA_DIR` to an absolute or relative path if data is not in `march-mania-ml/data`. `src/paths.py` resolves `DATA_DIR` from that variable.

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `src/paths.py` | Resolves project root and `DATA_DIR` / `get_data_dir()` from `MM_DATA_DIR`. |
| `src/io.py` | Loads CSVs from the data directory and checks that required competition files exist. |
| `src/elo.py` | End-of-season **Elo** snapshots for men’s and women’s games (home/away, K-factor, season regression). |
| `src/massey.py` | **Massey ordinals** and related features from `MMasseyOrdinals.csv` (men’s systems) and tournament linkage. |
| `src/features.py` | **Efficiency** and **four-factors** style metrics from detailed results; **matchup feature encoding** (`encode_matchups`, symmetric variant). |
| `src/calibration.py` | **Probability calibration** (e.g. isotonic / logistic wrappers) for tournament-style evaluation. |
| `src/nn_blend.py` | **PyTorch MLP** blend training and inference helpers; optional blend with other model scores. |
| `src/submission.py` | **Kaggle submission generation**: loads trained artifacts, builds features, applies calibrators / blends, writes prediction CSVs. |
| `src/inference.py` | **Feature store** construction mirroring submission-time features (useful for local checks or aligning with the API). |
| `src/eda.ipynb` | Exploratory analysis on competition tables. |
| `src/model.ipynb` | Modeling experiments and training workflows. |
| `src/hypetyune.ipynb` | Hyperparameter / tuning experiments. |
| `scripts/generate_teams_json.py` | Builds `teams.json` from `MTeams.csv` / `WTeams.csv` for frontend consumption (configure output path for your repo). |
| `scripts/generate_teams_json.mjs` | Node alternative for the same team export. |
| `tests/` | **pytest** suite: efficiency, four-factors, Elo, Massey, matchup encoding, calibration, NN blend, submission, inference. |
| `requirements.txt` | Full stack (Jupyter, PyTorch, XGBoost, etc.). |
| `requirements-dev.txt` | Dev/test tooling (e.g. pytest). |
| `pytest.ini` | Pytest configuration. |
| `.github/workflows/ci.yml` | CI running tests on Python 3.12. |

## Outputs consumed by other repos

After training and calibration, copy **artifacts** into the **backend** repository:

- `src/models/best_men.pkl`
- `src/models/best_women.pkl`
- `src/models/best_calibrator_men.joblib`
- `src/models/best_calibrator_women.joblib`

(Paths mirror the backend layout.) Optional blend weights / NN pickles follow the same `src/models/` convention used in your training code.

## Data

Download competition data from Kaggle into `./data` (or set `MM_DATA_DIR`). Never commit large CSVs unless your team policy allows it — use `.gitignore` and document where to fetch data.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` includes Jupyter, PyTorch, XGBoost, etc., for full experimentation.

## Tests

```bash
python -m pytest tests -q
```

CI runs the same command on Python 3.12 (`.github/workflows/ci.yml`).

## Workflow (high level)

1. Refresh `data/` from the competition.
2. Run notebooks or scripts to tune models and calibrators.
3. Write artifacts to `src/models/`.
4. Run `generate_submissions` / Stage-2 export paths in `src/submission.py` as needed for Kaggle.
5. Copy `src/models/*` to **march-mania-backend** for deployment.

## Related repositories

| Repo | Role |
| ---- | ---- |
| **march-mania-backend** | Serves trained models via FastAPI |
| **march-mania-web** | Next.js UI calling the backend |

## Contributing

Contributions are welcome: bug reports, documentation improvements, tests, and focused feature or modeling PRs. Please keep changes scoped, run `python -m pytest tests -q`, and match existing style and naming. If you add behavior, add or extend tests. Open an issue first for large design changes so we can align on approach.

## License

MIT — see `LICENSE`.
