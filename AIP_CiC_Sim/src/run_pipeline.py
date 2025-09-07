import argparse, os, sys, pathlib
import pandas as pd
import matplotlib.pyplot as plt

from cic_loader import load_filtered_corpus
from sim_aip_rsa import Params, make_lexicon, simulate_trial, wilson_ci

def find_csv(data_dir):
    for fn in sorted(os.listdir(data_dir)):
        if fn.lower().endswith(".csv"):
            return os.path.join(data_dir, fn)
    raise SystemExit(f"[error] No CSV found in {data_dir}")

def _salience_prior(C, beta=3.0):
    import numpy as np
    s = (C - 0.5)
    s = (s*s).sum(axis=1) ** 0.5
    s = s - s.max()
    p = np.exp(beta*s); p = p / p.sum()
    return p

def _ambiguity_score(C):
    import numpy as np
    from numpy.linalg import norm
    pairs = [(0,1),(0,2),(1,2)]
    d = [norm(C[i]-C[j]) for i,j in pairs]
    mind = float(min(d))
    return float(np.exp(-4.0 * mind))

def compute_trial_mods(C, flags, base_alpha, base_lam):
    """Return: prior_vec, len_adjust, alpha_eff, A6_on, kappa, lam."""
    A1 = flags.get("A1", True); A2 = flags.get("A2", True)
    A3 = flags.get("A3", True); A4 = flags.get("A4", True)
    A5 = flags.get("A5", True); A6 = flags.get("A6", True)

    prior_vec = None if A2 else _salience_prior(C, beta=3.0)
    len_adjust = 0.0 if A4 else 0.3 * _ambiguity_score(C)
    if A5:
        alpha_eff = base_alpha
    else:
        import numpy as np
        alpha_eff = float(max(1e-3, base_alpha * (1.0 + 0.3*np.random.default_rng().normal())))
    kappa = 0.3 if A1 else 0.0
    lam = base_lam if A3 else 0.0
    return prior_vec, len_adjust, alpha_eff, A6, kappa, lam

def bar_with_ci(df, path, order):
    import numpy as np
    df = df[df["setting"].isin(order)]
    if df.empty:
        return
    df = df.set_index("setting").loc[order].reset_index()
    x = np.arange(len(df))
    y = df["accuracy"].to_numpy()
    lo = df["acc_lo"].to_numpy(); hi = df["acc_hi"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])
    plt.figure(figsize=(7,4))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt='none', capsize=4)
    plt.xticks(x, df["setting"].tolist(), rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    import numpy as np  # local import to avoid UnboundLocalError in some environments

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default=None)
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--per_seed_n", type=int, default=100)
    ap.add_argument("--columns_json", default="columns_map.json")
    ap.add_argument("--condition_col", default="condition")
    ap.add_argument("--schema", choices=["cic_hsl"], default="cic_hsl")
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--lambda_cost", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eps_nonreg", type=float, default=0.2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_csv = args.data_csv or find_csv("data")
    if args.verbose:
        print("[run] data_csv:", data_csv)

    C_all, targets, conds = load_filtered_corpus(data_csv, args.columns_json, args.condition_col, verbose=args.verbose)

    out = pathlib.Path(args.out_dir); (out/"figures").mkdir(parents=True, exist_ok=True)

    params = Params(beta=args.beta, lam=args.lambda_cost, alpha=args.alpha, kappa=0.3, eps_nonreg=args.eps_nonreg)
    lex, lengths = make_lexicon(C_all, K=12, rng=params.rng)

    settings = [
        ("ALL_ON", dict(A1=True,A2=True,A3=True,A4=True,A5=True,A6=True)),
        ("A1_OFF", dict(A1=False,A2=True,A3=True,A4=True,A5=True,A6=True)),
        ("A2_OFF", dict(A1=True,A2=False,A3=True,A4=True,A5=True,A6=True)),
        ("A3_OFF", dict(A1=True,A2=True,A3=False,A4=True,A5=True,A6=True)),
        ("A4_OFF", dict(A1=True,A2=True,A3=True,A4=False,A5=True,A6=True)),
        ("A5_OFF", dict(A1=True,A2=True,A3=True,A4=True,A5=False,A6=True)),
        ("A6_OFF", dict(A1=True,A2=True,A3=True,A4=True,A5=True,A6=False)),
    ]

    rows = []
    rng = np.random.default_rng(0)
    idx_by_cond = {c: np.where(conds==c)[0] for c in ["close","far","split"]}
    if args.verbose:
        print("[run] condition sizes:", {k:int(v.size) for k,v in idx_by_cond.items()})

    for lab, flags in settings:
        if args.verbose:
            print("[run] setting:", lab)
        for cond in ["close","far","split"]:
            idxs = idx_by_cond[cond]
            if idxs.size == 0:
                if args.verbose:
                    print("  [skip] no data for", cond)
                continue
            accs=[]; lens=[]; risks=[]; costs=[]; igs=[]; ents=[]
            for seed in range(args.K):
                if args.verbose and (seed == 0 or (seed+1) % 5 == 0):
                    print("  [seed]", seed+1, "/", args.K)
                params.rng = np.random.default_rng(seed)
                chosen = rng.choice(idxs, size=args.per_seed_n, replace=True)
                for j in chosen:
                    prior_vec, len_adj, a_eff, A6_on, kappa, lam = compute_trial_mods(C_all[j], flags, params.alpha, args.lambda_cost)
                    params.kappa = kappa; params.lam = lam
                    a, L, r, cst, ig, ent = simulate_trial(C_all[j], int(targets[j]), lex, lengths, params,
                                                           prior_vec=prior_vec, len_adjust=len_adj, alpha_eff=a_eff, A6_softmax=A6_on)
                    accs.append(a); lens.append(L); risks.append(r); costs.append(cst); igs.append(ig); ents.append(ent)
            acc, lo, hi = wilson_ci(accs)
            successes = int(sum(accs))
            rows.append(dict(condition=cond, setting=lab, n=len(accs), successes=successes, accuracy=acc, acc_lo=lo, acc_hi=hi,
                             avg_len=float(pd.Series(lens).mean()), len_se=float(pd.Series(lens).std(ddof=1)/max(1,(len(lens))**0.5)),
                             avg_risk=float(pd.Series(risks).mean()), avg_cost=float(pd.Series(costs).mean()),
                             avg_IG=float(pd.Series(igs).mean()), ig_se=float(pd.Series(igs).std(ddof=1)/max(1,(len(igs))**0.5)),
                             avg_entropy=float(pd.Series(ents).mean())))

    summary = pd.DataFrame(rows)
    sum_path = out/"figures"/"summary.csv"
    summary.to_csv(sum_path, index=False)
    if args.verbose: print("[out] wrote", sum_path)

    # A1..A5 per condition
    order = ["ALL_ON","A1_OFF","A2_OFF","A3_OFF","A4_OFF","A5_OFF"]
    for cond in ["close","far","split"]:
        dfc = summary[summary["condition"]==cond]
        if not dfc.empty:
            path = out/"figures"/f"accuracy_A1toA5_{cond}.png"
            bar_with_ci(dfc, path, order=order)
            if args.verbose: print("[out] wrote", path)

    
    # A6-only pooled: compare ALL_ON vs A6_OFF using pooled successes + Wilson CI
    sub = summary[summary["setting"].isin(["ALL_ON","A6_OFF"])].copy()
    if not sub.empty:
        pooled = sub.groupby("setting", as_index=False)[["successes","n"]].sum()
        # enforce order: ALL_ON (left), A6_OFF (right)
        order = ["ALL_ON", "A6_OFF"]
        pooled = pooled.set_index("setting").loc[order].reset_index()

        import numpy as _np
        def wilson_from_counts(succ, n, z=1.96):
            if n == 0: return _np.nan, _np.nan, _np.nan
            p = succ / n
            denom = 1 + z**2/n
            centre = p + z*z/(2*n)
            pm = z*_np.sqrt((p*(1-p) + z*z/(4*n))/n)
            lo = (centre - pm) / denom
            hi = (centre + pm) / denom
            return p, lo, hi

        stats = [wilson_from_counts(int(r.successes), int(r.n)) for _, r in pooled.iterrows()]
        p = _np.array([s[0] for s in stats]); lo = _np.array([s[1] for s in stats]); hi = _np.array([s[2] for s in stats])
        yerr = _np.vstack([p - lo, hi - p])

        xs = _np.arange(len(order))
        plt.figure(figsize=(5.5,4))
        plt.bar(xs, p)
        plt.errorbar(xs, p, yerr=yerr, fmt='none', capsize=4)
        plt.xticks(xs, order)
        plt.ylim(0.0, 1.0); plt.ylabel("Accuracy (pooled across conditions)")
        plt.title("A6 regular vs. non-regular choice")
        path2 = out/"figures"/"accuracy_A6_only.png"
        plt.tight_layout(); plt.savefig(path2, dpi=160); plt.close()
        if args.verbose: print("[out] wrote", path2)
    else:
        if args.verbose: print("[warn] No rows for A6-only plot.")
# Table
    tex_path = out/"table_ablation.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.verbose: print("[out] wrote", tex_path)

if __name__ == "__main__":
    main()
