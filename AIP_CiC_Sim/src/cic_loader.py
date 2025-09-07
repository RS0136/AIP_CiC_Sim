import json, pathlib
import numpy as np
import pandas as pd

def _norm_status(s):
    s = str(s).strip().lower()
    if s in {"target","t","gold","correct","selected","click","true","1","yes","y"}:
        return "target"
    if s in {"distr","distractor","d1","d2","other","alt","alt1","alt2","alternative","false","0","no","n","non-target","nontarget"}:
        return "other"
    return s

def _hsl_component_norm(h, s, l):
    H = float(h); S = float(s); L = float(l)
    if H > 2.0: H = (H % 360.0) / 360.0
    if S > 1.0: S = S / 100.0
    if L > 1.0: L = L / 100.0
    H = max(0.0, min(1.0, H)); S = max(0.0, min(1.0, S)); L = max(0.0, min(1.0, L))
    return H, S, L

def _hsl_to_rgb01(h, s, l):
    if s == 0.0:
        return (l, l, l)
    def hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    q = l*(1+s) if l < 0.5 else l + s - l*s
    p = 2*l - q
    r = hue2rgb(p, q, h + 1/3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1/3)
    return (r, g, b)

def infer_condition(C, targets):
    from numpy.linalg import norm
    N = C.shape[0]
    out = []
    for i in range(N):
        t = targets[i]
        others = [j for j in [0,1,2] if j!=t]
        dt = [norm(C[i,t]-C[i,oj]) for oj in others]
        if max(dt) < 0.2:
            out.append("close")
        elif min(dt) > 0.4:
            out.append("far")
        else:
            out.append("split")
    return np.array(out, dtype=object)

def load_filtered_corpus(csv_path, columns_map_path=None, condition_col="condition", verbose=False):
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"[load] {csv_path} rows={len(df)} cols={len(df.columns)}")
    if columns_map_path:
        mapping = json.loads(pathlib.Path(columns_map_path).read_text(encoding="utf-8"))
    else:
        mapping = {
            "clickH":"clickColH","clickS":"clickColS","clickL":"clickColL","clickStatus":"clickStatus",
            "alt1H":"alt1ColH","alt1S":"alt1ColS","alt1L":"alt1ColL","alt1Status":"alt1Status",
            "alt2H":"alt2ColH","alt2S":"alt2ColS","alt2L":"alt2ColL","alt2Status":"alt2Status"
        }
    for k,v in mapping.items():
        if v not in df.columns:
            raise ValueError(f"Required column '{v}' not found. Edit columns_map.json or pass --columns_json.")

    C_list = []; t_idx = []
    for _, row in df.iterrows():
        ch,cs,cl = _hsl_component_norm(row[mapping["clickH"]], row[mapping["clickS"]], row[mapping["clickL"]])
        a1h,a1s,a1l = _hsl_component_norm(row[mapping["alt1H"]], row[mapping["alt1S"]], row[mapping["alt1L"]])
        a2h,a2s,a2l = _hsl_component_norm(row[mapping["alt2H"]], row[mapping["alt2S"]], row[mapping["alt2L"]])
        click = np.array(_hsl_to_rgb01(ch,cs,cl), dtype=float)
        alt1  = np.array(_hsl_to_rgb01(a1h,a1s,a1l), dtype=float)
        alt2  = np.array(_hsl_to_rgb01(a2h,a2s,a2l), dtype=float)
        sc = _norm_status(row[mapping["clickStatus"]])
        s1 = _norm_status(row[mapping["alt1Status"]])
        s2 = _norm_status(row[mapping["alt2Status"]])
        if sc=="target":
            C = np.stack([click,alt1,alt2], axis=0); tgt=0
        elif s1=="target":
            C = np.stack([alt1,click,alt2], axis=0); tgt=0
        elif s2=="target":
            C = np.stack([alt2,click,alt1], axis=0); tgt=0
        else:
            continue
        C_list.append(C); t_idx.append(tgt)
    if not C_list:
        raise RuntimeError("No valid rows with target status found.")
    C = np.stack(C_list, axis=0)
    t_idx = np.array(t_idx, dtype=int)

    if condition_col in df.columns:
        conds = df[condition_col].astype(str).str.lower().values[:len(t_idx)]
    else:
        conds = infer_condition(C, t_idx)

    if verbose:
        from collections import Counter
        print("[load] contexts:", C.shape, "targets:", t_idx.shape)
        print("[load] conditions:", dict(Counter(conds.tolist())))
    return C, t_idx, conds
