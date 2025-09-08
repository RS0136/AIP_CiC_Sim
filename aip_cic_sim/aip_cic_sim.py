#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aip_cic_sim.py — Parallel & reproducible CiC simulation (AIP→RSA)
=================================================================
- 同一のモデル/アルゴリズム（AIP→RSA）で、再現性を保ったままシード並列化。
- 引数： --input (filteredCorpus.csv のあるディレクトリ), --output (出力先), --procs (並列プロセス数)
- 定数（SEEDS, グリッドなど）はコード内に固定。

出力：figures/figure1.png〜figure9.png、tables/table1.tex・table2.tex、metrics.json と CSV/Excel（metrics/）。
"""

# ---- BLAS の非決定性回避（NumPy import 前に設定） ----
import os as _os
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    if _k not in _os.environ:
        _os.environ[_k] = "1"

import os
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

from multiprocessing import get_context

# -------------------------------
# Constants (edit if needed)
# -------------------------------
SEEDS: int = 30
BETA_GRID: Tuple[float, ...] = (4.0, 8.0, 16.0)
LAMBDA_GRID: Tuple[float, ...] = (0.3, 0.6, 0.9)
LAMBDA_AMB_GRID: Tuple[float, ...] = (0.0, 0.3, 0.6)
EPSILON: float = 0.05
KAPPA_LITERAL: float = 8.0
TOP_UNIGRAMS_FOR_CANDIDATES: int = 100
RNG_FOR_BOOTSTRAP: int = 123
BOOTSTRAP_B: int = 1000
BATCH_SIZE: int = 2000  # trials per batch for vectorized compute

# -------------------------------
# Data structures
# -------------------------------

@dataclass
class AIPConfig:
    beta: float = BETA_GRID[1]
    lambda_len: float = LAMBDA_GRID[1]
    lambda_amb: float = 0.0
    kappa_literal: float = KAPPA_LITERAL
    epsilon: float = EPSILON
    nonuniform_prior: bool = False
    miscalibrated_semantics: bool = False
    precision_drift: bool = False
    state_dependent_cost: bool = False  # A3 off approximation

@dataclass
class TrialInput:
    v_unit: np.ndarray   # (3,3) RGB unit vectors per chip
    sat: np.ndarray      # (3,) saturation per chip
    val: np.ndarray      # (3,) value/brightness per chip
    target_index: int    # 0
    condition: str       # close/split/far/pooled

@dataclass
class CandidateCache:
    utterances: List[str]
    int_mult: np.ndarray   # (M,)
    w_sat: np.ndarray      # (M,)
    w_val: np.ndarray      # (M,)
    B: np.ndarray          # (M,3)
    length: np.ndarray     # (M,)

# Globals for worker processes
_G_TRIALS: List[TrialInput] = []
_G_CC: Optional[CandidateCache] = None

# -------------------------------
# I/O utilities
# -------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_table_as_tex(df: pd.DataFrame, path: str, caption: str, label: str) -> None:
    lines: List[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(fr"\caption{{{caption}}}")
    lines.append(fr"\label{{{label}}}")
    cols = list(df.columns)
    align = "l" + "r"*(len(cols)-1)
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join([str(c) for c in cols]) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join([str(x) for x in row.values]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + (z*z)/n
    centre = p + (z*z)/(2*n)
    adj = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return (max(0.0, low), min(1.0, high))

# -------------------------------
# Parsing & color helpers
# -------------------------------

def _norm(v: float, big: float) -> float:
    if pd.isna(v):
        return 0.0
    try:
        x = float(v)
    except Exception:
        return 0.0
    if big == 360.0:
        return (x % 360.0) / 360.0 if x > 1.0 else x
    else:
        return min(1.0, x / 100.0) if x > 1.0 else x

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    H = _norm(h, 360.0)
    S = _norm(s, 100.0)
    L = _norm(l, 100.0)
    r, g, b = colorsys.hls_to_rgb(H, L, S)
    return (float(r), float(g), float(b))

def color_features(rgb: Tuple[float, float, float]) -> Tuple[np.ndarray, float, float]:
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v3 = np.array([r, g, b], dtype=float)
    nrm = np.linalg.norm(v3) + 1e-12
    v_unit = v3 / nrm
    return v_unit, float(s), float(v)

BASE_COLORS = {
    "red":     np.array([1.0, 0.0, 0.0]),
    "green":   np.array([0.0, 1.0, 0.0]),
    "blue":    np.array([0.0, 0.0, 1.0]),
    "yellow":  np.array([1.0, 1.0, 0.0]),
    "purple":  np.array([1.0, 0.0, 1.0]),
    "orange":  np.array([1.0, 0.5, 0.0]),
    "pink":    np.array([1.0, 0.6, 0.7]),
    "brown":   np.array([0.6, 0.4, 0.2]),
    "black":   np.array([0.0, 0.0, 0.0]),
    "white":   np.array([1.0, 1.0, 1.0]),
    "gray":    np.array([0.5, 0.5, 0.5]),
    "grey":    np.array([0.5, 0.5, 0.5]),
}
BASE_UNITS = {k: (v / (np.linalg.norm(v) + 1e-12)) for k, v in BASE_COLORS.items()}

MODIFIERS = {
    "light": ("value", +1.0),
    "dark":  ("value", -1.0),
    "bright":("saturation", +1.0),
    "pale":  ("saturation", -1.0),
    "saturated": ("saturation", +1.0),
    "dull":  ("saturation", -1.0),
    "very":  ("intensify", +1.0),
}

def tokenize(s: str) -> List[str]:
    return [t.strip().lower() for t in str(s).split() if t.strip()]

# -------------------------------
# Candidate processing (cached)
# -------------------------------

@dataclass
class CandidateCache:
    utterances: List[str]
    int_mult: np.ndarray
    w_sat: np.ndarray
    w_val: np.ndarray
    B: np.ndarray
    length: np.ndarray

def build_vocab(df: pd.DataFrame) -> Counter:
    vocab: Counter = Counter()
    for s in df["contents"].astype(str).values:
        for tok in tokenize(s):
            vocab[tok] += 1
    return vocab

def utterance_candidates_from_vocab(vocab: Counter) -> List[str]:
    bases = [w for w in vocab if w in BASE_COLORS] or ["red","blue","green","yellow","orange","purple","pink","brown","black","white","gray"]
    bases = sorted(bases)[:8]
    modifiers = ["", "light", "dark", "bright", "pale", "very"]
    cands = set()
    for b in bases:
        cands.add(b)
        for m in modifiers:
            if m:
                cands.add(f"{m} {b}")
    for w,_ in vocab.most_common(TOP_UNIGRAMS_FOR_CANDIDATES):
        if all(ch.isalpha() or ch=='-' for ch in w) and 2 <= len(w) <= 12:
            cands.add(w.lower())
    return sorted(cands)

def preprocess_candidates(candidates: List[str]) -> CandidateCache:
    M = len(candidates)
    int_mult = np.ones(M, dtype=float)
    w_sat = np.zeros(M, dtype=float)
    w_val = np.zeros(M, dtype=float)
    B = np.zeros((M, 3), dtype=float)
    length = np.zeros(M, dtype=float)
    for i, u in enumerate(candidates):
        toks = tokenize(u)
        length[i] = float(len(toks))
        n_very = sum(1 for t in toks if t == "very")
        int_mult[i] = 1.0 + 0.25 * n_very
        for t in toks:
            if t == "very":
                continue
            if t in BASE_UNITS:
                B[i, :] += BASE_UNITS[t]
            elif t in MODIFIERS:
                kind, sign = MODIFIERS[t]
                if kind == "saturation":
                    w_sat[i] += sign
                elif kind == "value":
                    w_val[i] += sign
    return CandidateCache(candidates, int_mult, w_sat, w_val, B, length)

# -------------------------------
# Dataset loading and trial preprocessing
# -------------------------------

def load_cic_filtered(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["condition","clickColH","clickColS","clickColL",
                "alt1ColH","alt1ColS","alt1ColL",
                "alt2ColH","alt2ColS","alt2ColL",
                "contents"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["condition"] = df["condition"].astype(str).str.lower().map({"close":"close","split":"split","far":"far"}).fillna("pooled")
    for c in ["clickColH","clickColS","clickColL","alt1ColH","alt1ColS","alt1ColL","alt2ColH","alt2ColS","alt2ColL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@dataclass
class TrialInput:
    v_unit: np.ndarray
    sat: np.ndarray
    val: np.ndarray
    target_index: int
    condition: str

def trials_from_df(df: pd.DataFrame) -> List[TrialInput]:
    trials: List[TrialInput] = []
    for _, row in df.iterrows():
        try:
            tgt = hsl_to_rgb(row["clickColH"], row["clickColS"], row["clickColL"])
            d1  = hsl_to_rgb(row["alt1ColH"], row["alt1ColS"], row["alt1ColL"])
            d2  = hsl_to_rgb(row["alt2ColH"], row["alt2ColS"], row["alt2ColL"])
        except Exception:
            continue
        v1, s1, vval1 = color_features(tgt)
        v2, s2, vval2 = color_features(d1)
        v3, s3, vval3 = color_features(d2)
        V = np.stack([v1, v2, v3], axis=0)
        S = np.array([s1, s2, s3], dtype=float)
        Vv= np.array([vval1, vval2, vval3], dtype=float)
        trials.append(TrialInput(V, S, Vv, target_index=0, condition=str(row["condition"])))
    return trials

# -------------------------------
# Speaker/utilities (vectorized core)
# -------------------------------

def ambiguity_penalty(P: np.ndarray) -> np.ndarray:
    H = -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=1)
    Hmax = math.log(3.0)
    return H / Hmax

def information_gain(prior: np.ndarray, post: np.ndarray) -> float:
    Hp = -np.sum(prior * np.log(np.clip(prior, 1e-12, 1.0)))
    Hq = -np.sum(post  * np.log(np.clip(post,  1e-12, 1.0)))
    return float(Hp - Hq)

def nonuniform_prior_from_val(vals: np.ndarray) -> np.ndarray:
    pr = vals + 1e-6
    pr = pr / np.sum(pr)
    return pr

def utility_matrix_for_trial(tr: TrialInput, CC: CandidateCache, cfg: "AIPConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sat = tr.sat; val = tr.val; VU = tr.v_unit

    base0 = CC.B @ VU[0]; base1 = CC.B @ VU[1]; base2 = CC.B @ VU[2]
    mod_sat0 = CC.w_sat * sat[0]; mod_sat1 = CC.w_sat * sat[1]; mod_sat2 = CC.w_sat * sat[2]
    mod_val0 = CC.w_val * val[0]; mod_val1 = CC.w_val * val[1]; mod_val2 = CC.w_val * val[2]

    S0 = CC.int_mult * (mod_sat0 + mod_val0 + base0)
    S1 = CC.int_mult * (mod_sat1 + mod_val1 + base1)
    S2 = CC.int_mult * (mod_sat2 + mod_val2 + base2)
    S = np.stack([S0, S1, S2], axis=1)

    X = cfg.kappa_literal * S
    X = X - np.max(X, axis=1, keepdims=True)
    P = np.exp(X); P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

    Amb = ambiguity_penalty(P)
    if cfg.state_dependent_cost:
        length_pen = CC.length * (1.0 + 0.5 * Amb)
    else:
        length_pen = CC.length

    U = np.log(P[:, 0] + 1e-12) - cfg.lambda_len * length_pen - cfg.lambda_amb * Amb
    return U, P[:, 0], P

def sample_softmax_from_util(U: np.ndarray, beta: float, rng: random.Random) -> int:
    x = beta * (U - np.max(U))
    w = np.exp(x)
    z = float(np.sum(w))
    if not np.isfinite(z) or z <= 0.0:
        return int(np.argmax(U))
    idx = rng.choices(range(len(U)), weights=w, k=1)[0]
    return int(idx)

def sample_epsilon_greedy(U: np.ndarray, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return int(rng.randrange(len(U)))
    return int(np.argmax(U))

# -------------------------------
# Summarization (batched)
# -------------------------------

def summarize_trials(trials: List[TrialInput], CC: CandidateCache, cfg: "AIPConfig", picker: str, seed: int) -> Dict[str, Any]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if cfg.precision_drift:
        beta_effective = max(2.0, float(np_rng.lognormal(mean=math.log(cfg.beta), sigma=0.1)))
    else:
        beta_effective = cfg.beta

    acc = 0; n = 0
    lengths: List[float] = []
    IGs: List[float] = []
    per_cond: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"correct":0, "n":0, "lengths":[], "IGs": []})

    T = len(trials)
    for start in range(0, T, BATCH_SIZE):
        end = min(T, start + BATCH_SIZE)
        batch = trials[start:end]
        for tr in batch:
            U, P_t, P_full = utility_matrix_for_trial(tr, CC, cfg)
            if picker == "softmax":
                idx = sample_softmax_from_util(U, beta_effective, py_rng)
            else:
                idx = sample_epsilon_greedy(U, cfg.epsilon, py_rng)
            post_row = P_full[idx, :]
            pred_idx = int(np.argmax(post_row))
            correct = (pred_idx == tr.target_index)
            acc += int(correct); n += 1
            lengths.append(float(CC.length[idx]))
            prior = nonuniform_prior_from_val(tr.val) if cfg.nonuniform_prior else np.array([1/3,1/3,1/3], dtype=float)
            IGs.append(information_gain(prior, post_row))
            b = per_cond[tr.condition]
            b["correct"] += int(correct); b["n"] += 1; b["lengths"].append(float(CC.length[idx])); b["IGs"].append(IGs[-1])

    overall_acc = acc/n if n>0 else 0.0
    lo, hi = wilson_interval(acc, n)
    out = {"overall": {"Accuracy": overall_acc, "CI_low": lo, "CI_high": hi,
                       "Length_mean": float(np.mean(lengths)) if lengths else 0.0,
                       "IG_mean": float(np.mean(IGs)) if IGs else 0.0},
           "by_condition": {}}
    for c, stats in per_cond.items():
        nn = stats["n"]; ss = stats["correct"]
        acc_c = ss/nn if nn>0 else 0.0
        lo_c, hi_c = wilson_interval(ss, nn)
        out["by_condition"][c] = {"Accuracy": acc_c, "CI_low": lo_c, "CI_high": hi_c,
                                  "Length_mean": float(np.mean(stats["lengths"])) if stats["lengths"] else 0.0,
                                  "IG_mean": float(np.mean(stats["IGs"])) if stats["IGs"] else 0.0,
                                  "n": nn}
    return out

# -------------------------------
# Plot helpers
# -------------------------------

def plot_bars_with_se(names: List[str], means: List[float], ses: List[float], title: str, out_path: str, ylabel: str = "Accuracy") -> None:
    plt.figure(figsize=(9,4))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=ses, capsize=3)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------------
# Metrics export (CSV & Excel)
# -------------------------------

def write_metrics_tables(metrics: Dict[str, Any], output_dir: str) -> None:
    import pandas as pd
    metrics_dir = os.path.join(output_dir, "metrics"); os.makedirs(metrics_dir, exist_ok=True)

    abl = metrics.get("ablations", {})
    rows_overall = []; rows_bycond = []
    for setting, rec in abl.items():
        ovr = rec.get("overall", {})
        rows_overall.append({"setting": setting, **{
            "Accuracy_mean": ovr.get("Accuracy_mean", float("nan")),
            "Accuracy_se": ovr.get("Accuracy_se", float("nan")),
            "Length_mean": ovr.get("Length_mean", float("nan")),
            "IG_mean": ovr.get("IG_mean", float("nan")),
        }})
        for cond, cvals in rec.get("by_condition", {}).items():
            rows_bycond.append({"setting": setting, "condition": cond, **{
                "Accuracy_mean": cvals.get("Accuracy_mean", float("nan")),
                "Accuracy_se": cvals.get("Accuracy_se", float("nan")),
                "Length_mean": cvals.get("Length_mean", float("nan")),
                "IG_mean": cvals.get("IG_mean", float("nan")),
            }})
    df_abl_overall = pd.DataFrame(rows_overall); df_abl_bycond = pd.DataFrame(rows_bycond)
    df_abl_overall.to_csv(os.path.join(metrics_dir, "ablations_overall.csv"), index=False)
    df_abl_bycond.to_csv(os.path.join(metrics_dir, "ablations_by_condition.csv"), index=False)

    regs = metrics.get("regular_vs_eps", {})
    pd.DataFrame([{"variant": k, **v} for k,v in regs.items()]).to_csv(os.path.join(metrics_dir, "regular_vs_eps.csv"), index=False)

    pd.DataFrame(metrics.get("sensitivity", [])).to_csv(os.path.join(metrics_dir, "sensitivity.csv"), index=False)

    sm = metrics.get("seed_metrics", {})
    accs = sm.get("Accuracy", []); lens = sm.get("Length", [])
    pd.DataFrame({"seed": list(range(1, len(accs)+1)), "Accuracy": accs, "Length": lens}).to_csv(os.path.join(metrics_dir, "seed_metrics.csv"), index=False)

    boot = metrics.get("bootstrap", {})
    pd.DataFrame([{"condition": k, **v} for k,v in boot.items()]).to_csv(os.path.join(metrics_dir, "bootstrap.csv"), index=False)

    pd.DataFrame(metrics.get("amb_sensitivity", [])).to_csv(os.path.join(metrics_dir, "amb_sensitivity.csv"), index=False)

    try:
        with pd.ExcelWriter(os.path.join(metrics_dir, "metrics.xlsx")) as xw:
            df_abl_overall.to_excel(xw, sheet_name="ablations_overall", index=False)
            df_abl_bycond.to_excel(xw, sheet_name="ablations_by_condition", index=False)
            pd.DataFrame([{"variant": k, **v} for k,v in regs.items()]).to_excel(xw, sheet_name="regular_vs_eps", index=False)
            pd.DataFrame(metrics.get("sensitivity", [])).to_excel(xw, sheet_name="sensitivity", index=False)
            pd.DataFrame({"seed": list(range(1, len(accs)+1)), "Accuracy": accs, "Length": lens}).to_excel(xw, sheet_name="seed_metrics", index=False)
            pd.DataFrame([{"condition": k, **v} for k,v in boot.items()]).to_excel(xw, sheet_name="bootstrap", index=False)
            pd.DataFrame(metrics.get("amb_sensitivity", [])).to_excel(xw, sheet_name="amb_sensitivity", index=False)
    except Exception as e:
        print(f"[WARN] Could not write Excel workbook: {e}")

# -------------------------------
# Worker pool setup
# -------------------------------

def _init_pool(cc: CandidateCache, trials: List[TrialInput]) -> None:
    global _G_CC, _G_TRIALS
    _G_CC = cc
    _G_TRIALS = trials

def _worker_task(args: Tuple[int, Dict[str, Any], str]) -> Tuple[int, Dict[str, Any]]:
    seed, cfg_d, picker = args
    cfg = AIPConfig(**cfg_d)
    res = summarize_trials(_G_TRIALS, _G_CC, cfg, picker, seed)
    return seed, res

def _run_seeds_parallel(cfg: AIPConfig, picker: str, procs: int) -> List[Dict[str, Any]]:
    from multiprocessing import get_context
    ctx = get_context("spawn")
    cfg_d = cfg.__dict__.copy()
    tasks = [(s, cfg_d, picker) for s in range(1, SEEDS+1)]
    with ctx.Pool(processes=int(procs), initializer=_init_pool, initargs=(_G_CC, _G_TRIALS)) as pool:
        results = list(tqdm(pool.imap_unordered(_worker_task, tasks), total=len(tasks), desc=f"pool:{picker}"))
    results.sort(key=lambda x: x[0])
    return [r for (_, r) in results]

# -------------------------------
# Main pipeline
# -------------------------------

def run(input_dir: str, output_dir: str, procs: int) -> None:
    csv_path = os.path.join(input_dir, "filteredCorpus.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"filteredCorpus.csv not found in: {input_dir}")
    print(f"[LOAD] {csv_path}")
    df = load_cic_filtered(csv_path)
    print(f"[INFO] rows={len(df)}")

    trials = trials_from_df(df)
    print(f"[INFO] usable trials with full colors={len(trials)}")
    vocab = build_vocab(df)
    candidates = utterance_candidates_from_vocab(vocab)
    CC = preprocess_candidates(candidates)
    print(f"[INFO] candidate utterances={len(candidates)} (cached features)")

    _init_pool(CC, trials)

    figdir = os.path.join(output_dir, "figures"); ensure_dir(figdir)
    tbldir = os.path.join(output_dir, "tables"); ensure_dir(tbldir)
    metrics: Dict[str, Any] = {}

    print("[1/6] Ablations A1--A5...")
    ablations = {
        "baseline": AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0),
        "A2_off_nonuniform_prior": AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0, nonuniform_prior=True),
        "A3_off_state_dependent_cost": AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0, state_dependent_cost=True),
        "A4_off_precision_drift": AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0, precision_drift=True),
        "A5_off_miscalibrated_semantics": AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0, miscalibrated_semantics=True),
    }
    ablation_summaries: Dict[str, Any] = {}
    for name, cfg in ablations.items():
        seed_summ = _run_seeds_parallel(cfg, "softmax", procs)
        def agg_mean(key: str) -> Tuple[float, float]:
            vals = [x["overall"][key] for x in seed_summ]
            return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))
        ablation_summaries[name] = {
            "overall": {
                "Accuracy_mean": agg_mean("Accuracy")[0],
                "Accuracy_se": agg_mean("Accuracy")[1],
                "Length_mean": float(np.mean([x["overall"]["Length_mean"] for x in seed_summ])),
                "IG_mean": float(np.mean([x["overall"]["IG_mean"] for x in seed_summ])),
            },
            "by_condition": {}
        }
        conds = set().union(*[set(s["by_condition"].keys()) for s in seed_summ])
        for c in conds:
            accs = [s["by_condition"][c]["Accuracy"] for s in seed_summ if c in s["by_condition"]]
            lens = [s["by_condition"][c]["Length_mean"] for s in seed_summ if c in s["by_condition"]]
            IGs  = [s["by_condition"][c]["IG_mean"] for s in seed_summ if c in s["by_condition"]]
            if accs:
                ablation_summaries[name]["by_condition"][c] = {
                    "Accuracy_mean": float(np.mean(accs)),
                    "Accuracy_se": float(np.std(accs)/math.sqrt(len(accs))),
                    "Length_mean": float(np.mean(lens)),
                    "IG_mean": float(np.mean(IGs)),
                }
    metrics["ablations"] = ablation_summaries

    def plot_ablation_for(cond: str, fpath: str, title: str) -> None:
        names = list(ablations.keys())
        means = [ablation_summaries[n]["by_condition"].get(cond, ablation_summaries[n]["overall"])["Accuracy_mean"] for n in names]
        ses   = [ablation_summaries[n]["by_condition"].get(cond, ablation_summaries[n]["overall"]).get("Accuracy_se", 0.0) for n in names]
        plot_bars_with_se(names, means, ses, title, fpath, ylabel="Accuracy")
    plot_ablation_for("close", os.path.join(figdir, "figure1.png"), "Ablations A1–A5 (close)")
    plot_ablation_for("split", os.path.join(figdir, "figure2.png"), "Ablations A1–A5 (split)")
    plot_ablation_for("far",   os.path.join(figdir, "figure3.png"), "Ablations A1–A5 (far)")
    plot_ablation_for("pooled",os.path.join(figdir, "figure4.png"), "Ablations A1–A5 (pooled)")

    print("[2/6] Regular (softmax) vs non-regular (epsilon-greedy)...")
    regs: Dict[str, Dict[str, float]] = {}
    base_cfg = AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=0.0)
    accs = [s["overall"]["Accuracy"] for s in _run_seeds_parallel(base_cfg, "softmax", procs)]
    regs["softmax"] = {"Accuracy_mean": float(np.mean(accs)), "Accuracy_se": float(np.std(accs)/math.sqrt(len(accs)))}
    accs = [s["overall"]["Accuracy"] for s in _run_seeds_parallel(base_cfg, "epsilon", procs)]
    regs["epsilon_greedy"] = {"Accuracy_mean": float(np.mean(accs)), "Accuracy_se": float(np.std(accs)/math.sqrt(len(accs)))}
    metrics["regular_vs_eps"] = regs
    plot_bars_with_se(["softmax", "epsilon-greedy"],
                      [regs["softmax"]["Accuracy_mean"], regs["epsilon_greedy"]["Accuracy_mean"]],
                      [regs["softmax"]["Accuracy_se"], regs["epsilon_greedy"]["Accuracy_se"]],
                      "Regular (logit) vs. non-regular ($\\varepsilon$-greedy)",
                      os.path.join(figdir, "figure5.png"))

    print("[3/6] Sensitivity over (beta, lambda)...")
    sens: List[Dict[str, float]] = []
    for beta in BETA_GRID:
        for lam in LAMBDA_GRID:
            cfg = AIPConfig(beta=float(beta), lambda_len=float(lam), lambda_amb=0.0)
            seed_summ = _run_seeds_parallel(cfg, "softmax", procs)
            sens.append({"beta": float(beta), "lambda": float(lam),
                         "Accuracy_mean": float(np.mean([x["overall"]["Accuracy"] for x in seed_summ])),
                         "Accuracy_se": float(np.std([x["overall"]["Accuracy"] for x in seed_summ])/math.sqrt(len(seed_summ))),
                         "Length_mean": float(np.mean([x["overall"]["Length_mean"] for x in seed_summ]))})
    metrics["sensitivity"] = sens
    plt.figure(figsize=(7,4))
    lambdas = sorted(set(x["lambda"] for x in sens))
    for lam in lambdas:
        xs=[]; ys=[]; ys_se=[]
        for beta in sorted(set(x["beta"] for x in sens)):
            rec = next(r for r in sens if r["beta"]==beta and r["lambda"]==lam)
            xs.append(beta); ys.append(rec["Accuracy_mean"]); ys_se.append(rec["Accuracy_se"])
        plt.errorbar(xs, ys, yerr=ys_se, marker="o", label=f"lambda={lam}")
    plt.xlabel("beta (precision)"); plt.ylabel("Accuracy"); plt.title("Sensitivity over (beta, lambda)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "figure6.png"), dpi=200); plt.close()

    print("[4/6] Seed-wise distributions...")
    seed_summ = _run_seeds_parallel(base_cfg, "softmax", procs)
    accs = [s["overall"]["Accuracy"] for s in seed_summ]; lens = [s["overall"]["Length_mean"] for s in seed_summ]
    metrics["seed_metrics"] = {"Accuracy": accs, "Length": lens}
    plt.figure(figsize=(7,4)); plt.boxplot([accs, lens], labels=["Accuracy","Length"]); plt.title("By-seed distributions (box)")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "figure7.png"), dpi=200); plt.close()

    print("[5/6] Stratified bootstrap...")
    rng = np.random.default_rng(RNG_FOR_BOOTSTRAP)
    once = []
    for tr in tqdm(_G_TRIALS, desc="bootstrap pass"):
        cfg0 = base_cfg
        U, P_t, P_full = utility_matrix_for_trial(tr, _G_CC, cfg0)
        post = P_full[np.argmax(U)]
        once.append((tr.condition, int(np.argmax(post) == tr.target_index)))
    conditions = sorted({c for c,_ in once})
    by_cond = {c: [y for (cc,y) in once if cc==c] for c in conditions}
    B = BOOTSTRAP_B
    boot = {c: [] for c in conditions}
    for b in tqdm(range(B), desc="bootstrap resamples"):
        for c in conditions:
            arr = by_cond[c]
            if not arr:
                boot[c].append(0.0); continue
            samp = rng.choice(arr, size=len(arr), replace=True)
            boot[c].append(float(np.mean(samp)))
    boot_summ = {c: {"mean": float(np.mean(v)), "low": float(np.quantile(v,0.025)), "high": float(np.quantile(v,0.975))} for c,v in boot.items()}
    metrics["bootstrap"] = boot_summ
    plt.figure(figsize=(6,4))
    xs = np.arange(len(conditions))
    means = [boot_summ[c]["mean"] for c in conditions]
    lows  = [boot_summ[c]["mean"] - boot_summ[c]["low"]  for c in conditions]
    highs = [boot_summ[c]["high"] - boot_summ[c]["mean"] for c in conditions]
    plt.errorbar(xs, means, yerr=[lows, highs], fmt="o")
    plt.xticks(xs, conditions); plt.ylabel("Accuracy"); plt.title("Stratified bootstrap CIs")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "figure8.png"), dpi=200); plt.close()

    print("[6/6] Ambiguity penalty sensitivity...")
    amb_res = []
    for lam_amb in LAMBDA_AMB_GRID:
        cfg = AIPConfig(beta=BETA_GRID[1], lambda_len=LAMBDA_GRID[1], lambda_amb=float(lam_amb))
        seed_summ = _run_seeds_parallel(cfg, "softmax", procs)
        amb_res.append({"lambda_amb": float(lam_amb),
                        "Accuracy_mean": float(np.mean([x["overall"]["Accuracy"] for x in seed_summ])),
                        "Accuracy_se": float(np.std([x["overall"]["Accuracy"] for x in seed_summ])/math.sqrt(len(seed_summ))),
                        "Length_mean": float(np.mean([x["overall"]["Length_mean"] for x in seed_summ]))})
    metrics["amb_sensitivity"] = amb_res
    plt.figure(figsize=(7,4))
    xs = [r["lambda_amb"] for r in amb_res]; ys = [r["Accuracy_mean"] for r in amb_res]; yse = [r["Accuracy_se"] for r in amb_res]
    plt.errorbar(xs, ys, yerr=yse, marker="o"); plt.xlabel("lambda_amb"); plt.ylabel("Accuracy"); plt.title("Ambiguity penalty sensitivity")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "figure9.png"), dpi=200); plt.close()

    t1 = pd.DataFrame([
        {"Setting": name,
         "Accuracy": f"{ablation_summaries[name]['overall']['Accuracy_mean']:.3f}",
         "SE": f"{ablation_summaries[name]['overall']['Accuracy_se']:.3f}",
         "Length": f"{ablation_summaries[name]['overall']['Length_mean']:.3f}",
         "IG": f"{ablation_summaries[name]['overall']['IG_mean']:.3f}"}
        for name in ablations.keys()
    ])
    save_table_as_tex(t1, os.path.join(tbldir, "table1.tex"),
        caption="Ablations A1--A5 (pooled). Accuracy (mean$\\pm$SE), Length, and IG.",
        label="tab:ablation")

    t2 = pd.DataFrame([
        {"beta": r["beta"], "lambda": r["lambda"],
         "Accuracy": f"{r['Accuracy_mean']:.3f}",
         "SE": f"{r['Accuracy_se']:.3f}",
         "Length": f"{r['Length_mean']:.3f}"}
        for r in metrics["sensitivity"]
    ])
    save_table_as_tex(t2, os.path.join(tbldir, "table2.tex"),
        caption="Sensitivity over $(\\beta,\\lambda)$. Accuracy (mean$\\pm$SE) and Length.",
        label="tab:sensitivity")

    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_metrics_tables(metrics, output_dir)

    print(f"[DONE] Figures: {figdir}")
    print(f"[DONE] Tables:  {tbldir}")
    print(f"[DONE] Metrics: {os.path.join(output_dir, 'metrics.json')}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory containing filteredCorpus.csv")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--procs", type=int, default=1, help="Number of worker processes (physical cores)")
    args = ap.parse_args()
    run(args.input, args.output, max(1, int(args.procs)))

if __name__ == "__main__":
    main()
