# Hub data contract (filter → Sheet → Streamlit)

End-to-end path for every hub (PEN353, PEN364, …):

```
JMS export (https://jms.jtexpress.my)
  → python -m hub_filter --hub pen353   (or pen364)
  → Template JMR–shaped .xlsx
  → upload / sync to that hub’s Google Sheet
  → Streamlit Cloud app (same deploy/* branch, different secrets.gsheet_url)
```

## Rules

1. **One codebase** — `deploy/dispatcher` and `deploy/management` only (role split). Do **not** create a Git branch per hub.
2. **One Streamlit app per hub×role** — isolate with Secrets (`gsheet_url` or Postgres URL), same branch.
3. **Filter output columns** = Template JMR / `hub_filter.contract.OUTPUT_SHEETS` = `sheet_schema.SHEET_COLUMNS`.
4. **QR Order** is not a sheet — use Pickup `order_source` = `JTD QR` (RM1.80 commission).
5. **LDR** money = `penalty`; period date = `pushed_time` (ignore `penalty_amount` in calc).
6. **COD** money = `penalty`.
7. **Fake Attempt** keep: `date`, `waybill_number`, `dispatcher_id`, `dispatcher_name` (RM2 × rows).
8. **No Outbound Scan** keep: `waybill_number`, `dispatcher_id` (RM3 × unique AWB). No date → export only the payroll month into the tab.

## Run filter

From `payout-calculator/`:

```bash
python -m hub_filter --hub pen353
python -m hub_filter --hub pen364
```

Configs: `hubs/pen353.json`, `hubs/pen364.json` (edit `file` / `output_file` / `dispatcher_ids` per month).

Legacy scripts `../filter-353.py` and `../filter-364.py` call the same module.

## Add a new hub

1. Copy `hubs/pen364.json` → `hubs/penXXX.json`; set IDs and Excel paths.
2. Run `python -m hub_filter --hub penXXX`.
3. Create Google Sheet from Template JMR; paste filtered tabs.
4. Deploy a **new** Streamlit Cloud app → branch `deploy/dispatcher` (or management) → set Secrets `gsheet_url` for that sheet.
5. No code fork required.

## Reference template

Local blank headers:  
`/Users/UMARI.ZULKIFLI/Documents/Jemari Ventures/Worksheet/Template JMR.xlsx`
