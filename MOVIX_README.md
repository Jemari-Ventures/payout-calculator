# Movix IC Courier Payout

> **Canonical repo:** this Movix IC app now lives at [Jemari-Ventures/movix-ic-payout](https://github.com/Jemari-Ventures/movix-ic-payout). Prefer that repository for new work; this copy is retained for reference on the `movix-dispatcher` branch.


Standalone Streamlit app for **Movix Logistics OS Karangan (OKAR)** courier payment.

This is **separate from** the legacy hub dispatcher (`app.py`) and management (`management.py`) calculators. Do not mix configs or sheets between them.

## Rate card (from Movix courier payment poster)

| Component | Rule |
|-----------|------|
| Commission | `0–5.00 kg` → RM1.00 · `5.01–20.00 kg` → RM1.80 · `>20 kg` → RM3.00 per parcel |
| Basic salary | RM1,700 if ≥26 working days **and** avg ≥50 parcels/day |
| Attendance | RM200 with the same volume gate; MC deducts RM100/day; ≥2 MC days → no allowance |
| Absent | RM30/day unauthorized absence |
| If volume gate fails | Commission only (no basic, no attendance) |

Billable rows: service types starting with `Del - FWD` (D2D / BSC). Failed deliveries, RTS, and pickup rows are excluded.

## Input

Excel workbook, **Sheet1** only, with columns:

- Service Date
- Service Type
- LM Type (optional)
- Shipment ID
- Chargeable Weight
- Driver ID
- Driver Name
- TIER (optional; informational — rates come from weight)

## Run

```bash
cd /path/to/payout-calculator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-movix.txt
streamlit run movix_dispatcher/app.py
```

Upload the IC Payment `.xlsx` in the sidebar. Use **Batch payout** for all drivers or **Single driver** for detail + CSV export.

## Config

Rates live in [`config_movix.json`](config_movix.json). Change amounts / bands there without touching code.

## Tests

```bash
pip install -r requirements-movix.txt
PYTHONPATH=. pytest tests/test_movix_*.py -q
```

If `~/Documents/IC Payment.xlsx` is present, an extra live ingest test runs automatically.

## Package layout

```
movix_dispatcher/
  app.py           # Streamlit UI
  calculator.py    # Payout engine
  config.py        # Rate-card loader
  eligibility.py   # Basic + attendance gates
  ingest.py        # Sheet1 load / normalize
  rates.py         # Weight → commission
  schema.py        # Column contract
config_movix.json
requirements-movix.txt
```
