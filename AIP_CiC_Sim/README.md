# AIP→RSA Simulator for CiC (`filteredCorpus.csv`)

## Quick start
1) Put `filteredCorpus.csv` into `data/`.
2) Run:
```bash
pip install -r requirements.txt
python src/run_pipeline.py --schema cic_hsl --columns_json columns_map.json --condition_col condition --K 5 --per_seed_n 100 --verbose
```
(増やす場合は `--K 30 --per_seed_n 200` などに。)

### What you get
- `output/figures/summary.csv` — per (setting, condition) means/CIs
- `output/figures/accuracy_A1toA5_close.png`, `..._far.png`, `..._split.png`
- `output/figures/accuracy_A6_only.png`
- `output/table_ablation.tex` — Table 1 LaTeX

## Input schema (filteredCorpus.csv)
- CiC ネイティブ列（推奨）:
  - `clickColH/S/L`, `alt1ColH/S/L`, `alt2ColH/S/L`
  - `clickStatus`, `alt1Status`, `alt2Status` （`target` / `distractor` 等）
  - `condition`（close / far / split） ※無ければ自動推定
- この repo は HSL→RGB(0–1) に自動変換します。

## Column mapping
標準の列名なら同梱の `columns_map.json` をそのまま使えます。
```bash
python src/run_pipeline.py --schema cic_hsl --columns_json columns_map.json
```
列名が微妙に違うときは `columns_map.json` の値を編集してください。

## Notes
- 第三者データ（CiC）は再配布しません。公式配布元から取得して `data/` に置いてください。
- まずは `--K 5 --per_seed_n 100` で動作確認し、問題なければ増やしてください。


### Settings now included
- `ALL_ON`, `A1_OFF`, `A2_OFF`, `A3_OFF`, `A4_OFF`, `A5_OFF`, `A6_OFF`。
- 図 `accuracy_A1toA5_*.png` は `ALL_ON` と `A1..A5_OFF` を並べ、`A6_OFF` は別図（`accuracy_A6_only.png`）。
