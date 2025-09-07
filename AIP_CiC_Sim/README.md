# AIP→RSA Simulator (CiC `filteredCorpus.csv`) — v8c

## How to run
1. Put `filteredCorpus.csv` into `data/` (CiC native columns).
2. Install & run:
```bash
pip install -r requirements.txt
python src/run_pipeline.py --schema cic_hsl --columns_json columns_map.json --condition_col condition --K 5 --per_seed_n 100 --verbose
```
Increase to `--K 30 --per_seed_n 200` for paper numbers.

## Outputs
- `output/figures/summary.csv` (now includes `successes` counts)
- `output/figures/accuracy_A1toA5_close.png` (also `far`, `split`)
- `output/figures/accuracy_A6_only.png` (fixed: compares **ALL_ON vs A6_OFF** pooled with Wilson 95% CI)
- `output/table_ablation.tex`

## Notes
- We do **not** redistribute CiC. Obtain from the official source and place CSV in `data/`.
- Column names can be remapped via `columns_map.json`.
