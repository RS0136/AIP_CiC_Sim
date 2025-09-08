# aip_cic_sim.py — README

**AIP→RSA CiC シミュレーション（並列・再現性担保版）**  
本スクリプトは、Free Energy Principle/Active Inference に基づく話者モデルを用いて、Rational Speech Acts (RSA) の特殊解としての振る舞いを **CiC (Colors in Context)** 実験データで再現・評価するための **完全再現性**・**並列化対応**の実装です。

---

## 特長

- **手法は既存コードと同一**：尤度（literal listener）、効用、選択規則（softmax/ε-greedy）、グリッド（β・λ）、A1–A5 アブレーション等は同一。
- **並列化しても再現性**：`multiprocessing` の **spawn**、BLAS の単スレ化、**seed ごと独立集計→決定的順序で集約**により、`--procs` の値に依らず同一結果。
- **進捗可視化**：tqdm でステージ/プールごとの進捗を表示。
- **出力の互換性**：Figures (1–9), Tables (1–2), `metrics.json`、さらに **CSV 群と Excel** を自動生成。

---

## 必要環境

- Python **3.9+**（推奨 3.10–3.12）
- 主要ライブラリ
  - `numpy`, `pandas`, `matplotlib`, `tqdm`
- OS: Linux / macOS / Windows（WSL 含む）

> **BLAS の非決定性回避**のため、スクリプト先頭で `OMP_NUM_THREADS=1` などの環境変数を自動設定します。既に設定済みの環境では上書きしません。

---

## 入力データ

`--input` で指定するディレクトリに **`filteredCorpus.csv`** を置いてください。必須カラム：

```
condition, clickColH, clickColS, clickColL,
alt1ColH, alt1ColS, alt1ColL,
alt2ColH, alt2ColS, alt2ColL,
contents
```

- `condition` は `close/split/far` を想定（その他は `pooled` として扱います）。
- 角度・百分率は **HSL** 想定（度・% を 0–360, 0–100 として与えても自動正規化）。
- `contents` は発話テキスト。語彙から **候補発話** を構築します（上位出現 unigram、色語＋修飾語を自動生成）。

> 列名が異なる場合は、事前に列名を上記に揃えてください。

---

## 使い方

```bash
python aip_cic_sim.py \
  --input /path/to/dir_with_filteredCorpus_csv \
  --output /path/to/output_dir \
  --procs 14      # 物理コア数。環境に合わせて変更
```

- `--procs`：ワーカープロセス数（推奨＝物理コア数）。未指定時は 1。
- 実行中、`[1/6] …` 〜 `[6/6] …` のステージ進捗と、`pool:softmax` 等のプール進捗が表示されます。

---

## 出力

```
output/
  figures/
    figure1.png ... figure9.png
  tables/
    table1.tex, table2.tex
  metrics.json
  metrics/
    ablations_overall.csv
    ablations_by_condition.csv
    regular_vs_eps.csv
    sensitivity.csv
    seed_metrics.csv
    bootstrap.csv
    amb_sensitivity.csv
    metrics.xlsx   # 環境により作成不可のことあり
```

- **Figures 1–4**：A1–A5 アブレーション（close/split/far/pooled）
- **Figure 5**：regular (softmax) vs **ε-greedy**
- **Figure 6**：感度（β, λ）
- **Figure 7**：seed 分布（箱ひげ）
- **Figure 8**：層化ブートストラップ CI
- **Figure 9**：曖昧性ペナルティ λ_amb 感度  
- **Table 1**：A1–A5 (pooled) の要約  
- **Table 2**：感度（β, λ）の要約

---

## 再現性について

- プロセス開始方式に **spawn** を採用し、各 seed は独立ワーカーで処理。
- BLAS/LAPACK を **単スレッド**に固定（`OMP_NUM_THREADS=1` 等）。
- 乱数は **seed ごとに固定**。集約は **seed 昇順**で行い、加算順序の差による丸めゆらぎを抑制。
- 検証手順：同じコマンドを 2 回実行し、以下を比較
  - `diff output/metrics.json output2/metrics.json`
  - 主要 CSV（`ablations_overall.csv` 等）の一致

---

## パフォーマンスのヒント

- `--procs` は **物理コア数**を目安に。ハイパースレッディングに過度に依存しないこと。
- I/O がボトルネックの場合、出力先をローカル SSD に。
- 長時間実行時は `tmux` / `screen` / `nohup` の利用を推奨。

---

## よくある質問（FAQ）

**Q. 列が足りない/名前が違うと言われる**  
A. CSV の列名を上記の必須名に揃えてください。事前リネームで解決します。

**Q. tqdm が表示されない**  
A. `tqdm` 未導入の場合は標準イテレータにフォールバックします（進捗が出ません）。`pip install tqdm` で導入可。

**Q. Figures が真っ白/フォント警告が出る**  
A. `matplotlib` のバックエンドやフォント周りです。GUI 環境が無い場合も保存は可能です。`MPLBACKEND=Agg` を設定して再実行してください。

**Q. metrics.xlsx が出ない**  
A. `openpyxl` 等の writer が無い環境では自動でスキップします（CSV は生成されます）。

---

## 定数の変更

コード冒頭の定数を編集できます（結果の比較可能性のため既定値を推奨）。

```python
SEEDS = 30
BETA_GRID = (4.0, 8.0, 16.0)
LAMBDA_GRID = (0.3, 0.6, 0.9)
LAMBDA_AMB_GRID = (0.0, 0.3, 0.6)
EPSILON = 0.05
KAPPA_LITERAL = 8.0
TOP_UNIGRAMS_FOR_CANDIDATES = 100
BOOTSTRAP_B = 1000
```

---

## 参考

- FEP/AIF の枠組みの中で RSA を特殊解として位置づけ、A1–A6 の逸脱がどの挙動差を生むかを検証できるよう設計されています。
- 既存プレプリントの図表（Fig.1–9 / Table 1–2）と同一フォーマットの出力を生成します。

---

## ライセンス

この README はあなたのプロジェクト内で自由に利用・改変して構いません。
