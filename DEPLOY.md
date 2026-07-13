# Streamlit Cloud deployment

This repo runs **two** Streamlit apps from the same codebase:

| App | Entry file | Branch | Dependencies |
|-----|------------|--------|--------------|
| **Dispatcher payout** | `app.py` | `deploy/dispatcher` | Lightweight (no Prophet/plotly) |
| **Management batch** | `management.py` | `deploy/management` | Full (`prophet`, `plotly`, `sqlalchemy`, …) |

Shared code (`penalty_common.py`, `sheet_schema.py`, `config.json`, etc.) lives on both branches. Only `requirements.txt` differs per deploy branch.

---

## Source-of-truth requirement files

| File | Used by |
|------|---------|
| `requirements-dispatcher.txt` | `app.py` / dispatcher Cloud app |
| `requirements-management.txt` | `management.py` / management Cloud app |
| `requirements.txt` | **Generated** — do not edit by hand on deploy branches; run the sync script |

```bash
./scripts/sync-requirements.sh dispatcher   # app
./scripts/sync-requirements.sh management   # management
```

---

## Streamlit Community Cloud settings

Create **two apps** in [share.streamlit.io](https://share.streamlit.io) pointing at the same GitHub repo.

### Dispatcher app

| Setting | Value |
|---------|--------|
| **Branch** | `deploy/dispatcher` |
| **Main file path** | `app.py` *(or `payout-calculator/app.py` if repo root is parent)* |
| **Python version** | `3.11` via `packages.txt` |

### Management app

| Setting | Value |
|---------|--------|
| **Branch** | `deploy/management` |
| **Main file path** | `management.py` *(or `payout-calculator/management.py`)* |
| **Python version** | `3.11` via `packages.txt` |

Both apps share:

- `.streamlit/config.toml` (repo root)
- `.streamlit/secrets.toml` (not committed — configure in Cloud **Secrets**)

---

## Development workflow

1. **Develop on `main` or `mgmt`** (feature branches → merge there).
2. **Merge into both deploy branches** when ready to release:

```bash
# From your dev branch (e.g. mgmt or main)
git fetch origin

# Dispatcher
git checkout deploy/dispatcher
git merge mgmt   # or main
./scripts/sync-requirements.sh dispatcher
git add requirements.txt
git commit -m "deploy: sync dispatcher requirements"  # skip if unchanged
git push -u origin deploy/dispatcher

# Management
git checkout deploy/management
git merge mgmt   # or main
./scripts/sync-requirements.sh management
git add requirements.txt
git commit -m "deploy: sync management requirements"  # skip if unchanged
git push -u origin deploy/management

git checkout mgmt
```

3. Streamlit Cloud redeploys automatically on push.

---

## Local run

```bash
python -m venv .venv && source .venv/bin/activate

# Dispatcher
./scripts/sync-requirements.sh dispatcher
pip install -r requirements.txt
streamlit run app.py

# Management
./scripts/sync-requirements.sh management
pip install -r requirements.txt
streamlit run management.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: prophet` on management | On `deploy/management`, run sync script and redeploy |
| App crashes / OOM on dispatcher | Ensure Cloud app uses **`deploy/dispatcher`**, not management branch |
| Stale sheet data | Use **Clear cache & reload** in the dispatcher app sidebar |
| `width` / `use_container_width` errors | `streamlit_compat.py` handles version differences |
| Merge conflict markers in `requirements.txt` | Run `./scripts/sync-requirements.sh management` (or `dispatcher`), commit, redeploy |

**After every merge into a deploy branch**, always run the sync script — never commit `requirements.txt` with `<<<<<<<` markers.

---

## Branch diagram

```
main / mgmt  ──merge──►  deploy/dispatcher  (requirements-dispatcher.txt → requirements.txt)
              └──merge──►  deploy/management   (requirements-management.txt → requirements.txt)
```

Do **not** develop only on deploy branches — merge from `main`/`mgmt` regularly to avoid drift.
