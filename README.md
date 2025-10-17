Dispatcher Payout Calculator

Quick Streamlit app to filter rows by a dispatcher ID from an Excel file and compute payout.

Supports two payout modes:
- Per parcel: payout = unique waybills × rate_per_parcel
- Tiered daily: sum unique waybills per day, map to tier ranges, apply per-parcel rate

Prerequisites
- Python 3.9+

Setup
```bash
cd /Users/UMARI.ZULKIFLI/dispatcher-payout-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run
```bash
source .venv/bin/activate
streamlit run app.py
```

Usage
- Upload an Excel `.xlsx` file in the sidebar.
- If multiple sheets exist, pick the sheet.
- Choose the `Dispatcher ID` column.
- Select the `Waybill number` column (each non-empty waybill counts as one parcel).
- Choose the Payout mode:
  - Per parcel: enter the `Rate per parcel`.
  - Tiered daily: select the `Date column` (used for daily grouping) and define tiers in the editor with columns `min_inclusive`, `max_inclusive` (leave empty for open-ended), and `rate_per_day`.
- Select the dispatcher ID to filter.
- In Per parcel mode, the app computes `parcel_count × rate_per_parcel` and displays a `_payout` column and total payout. Download exports the filtered rows plus `_payout`.
- In Tiered daily mode, the app groups by date, counts unique waybills per day as `daily_parcels`, shows columns `__date`, `daily_parcels`, `tier`, `rate_per_parcel`, `payout_per_day` (calculated as `daily_parcels × rate_per_parcel`), and the total payout. Download exports the daily summary.

 Notes
 - Excel engine: openpyxl
 - Waybill column can be any text-like column; non-empty values are counted.
 - For Tiered daily, the date column will be coerced to date (no time component).
 - Default tiers: ≤60 → 0.95, 61–120 → 1.00, ≥121 → 1.10 (per parcel).
