import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

@dataclass
class Params:
    beta: float = 8.0          # inverse temperature (A6 ON → softmax)
    lam: float = 0.6           # length/cost pressure (A3)
    alpha: float = 1.0         # semantic sharpness for L0
    kappa: float = 0.0         # IG weight (A1 OFF → 0, ALL_ON → >0)
    eps_nonreg: float = 0.2    # epsilon for non-regular (A6 OFF)
    rng: np.random.Generator = np.random.default_rng(0)

def softmax(x, beta):
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(beta * x)
    return e / e.sum()

def make_lexicon(C, K=12, rng=None):
    import numpy as np
    # simple KMeans++ init on pooled swatches
    X = C.reshape(-1,3)
    if rng is None:
        rng = np.random.default_rng(0)
    centers = X[rng.choice(len(X), 1, replace=False)]
    while len(centers) < K:
        d2 = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        new = X[rng.choice(len(X), 1, p=probs)]
        centers = np.vstack([centers, new])
    # Lloyd few steps
    for _ in range(5):
        dists = np.stack([np.sum((X - c)**2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        for k in range(len(centers)):
            pts = X[labels==k]
            if len(pts)>0:
                centers[k] = pts.mean(axis=0)
    # length cost ~ prototype id complexity (toy)
    lengths = np.linspace(1.0, 2.0, len(centers))
    return centers, lengths

def l0_posterior(u_proto, C, alpha):
    import numpy as np
    # literal listener: prefer closer color to utterance prototype
    util = -np.array([norm(u_proto - C[j]) for j in range(3)])
    return softmax(util, beta=alpha)

def speaker_score(u_proto, C, t_idx, alpha, lam, u_len_eff, kappa, prior_t):
    import numpy as np
    # informativeness: log p_L0(t|u) - log p_prior(t)
    pL0 = l0_posterior(u_proto, C, alpha)
    info = np.log(pL0[t_idx] + 1e-9) - np.log(prior_t + 1e-12)
    # risk proxy: hinge on margin between target and best distractor under L0
    sorted_idx = np.argsort(-pL0)
    margin = pL0[sorted_idx[0]] - pL0[sorted_idx[1]]
    risk = margin  # higher is better
    # utility
    U = risk + kappa*info - lam*u_len_eff
    return U, pL0

def simulate_trial(C, t_idx, lexicon, lengths, params: Params, prior_vec=None, len_adjust=0.0, alpha_eff=None, A6_softmax=True):
    import numpy as np
    # choose utterance
    import numpy as np
    scores = []
    pL0s = []
    for u, L in zip(lexicon, lengths):
        use_alpha = params.alpha if alpha_eff is None else alpha_eff
        sc, pL0 = speaker_score(u, C, t_idx, use_alpha, params.lam, L + len_adjust, params.kappa, prior_t=(1/3 if prior_vec is None else prior_vec[t_idx]))
        scores.append(sc); pL0s.append(pL0)
    scores = np.array(scores); pL0s = np.stack(pL0s, axis=0)
    if A6_softmax:
        pu = softmax(scores, beta=params.beta)
        u_idx = params.rng.choice(len(lexicon), p=pu)
    else:
        if params.rng.random() < params.eps_nonreg:
            u_idx = params.rng.integers(len(lexicon))
        else:
            u_idx = int(np.argmax(scores))
    # listener chooses referent given utterance
    pL0 = pL0s[u_idx]
    if A6_softmax:
        pt = softmax(pL0, beta=params.beta)
        choice = params.rng.choice(3, p=pt)
    else:
        if params.rng.random() < params.eps_nonreg:
            choice = params.rng.integers(3)
        else:
            choice = int(np.argmax(pL0))
    acc = 1.0 if choice==t_idx else 0.0
    length = lengths[u_idx]
    ig = np.log(pL0[t_idx]+1e-9) - np.log(1/3)
    risk = np.sort(pL0)[-1] - np.sort(pL0)[-2]
    cost = length
    ent = -np.sum(pL0*np.log(pL0+1e-12))
    return acc, length, risk, cost, ig, ent

def wilson_ci(accs, z=1.96):
    import numpy as np
    n = len(accs)
    if n==0:
        return (np.nan, np.nan, np.nan)
    p = np.mean(accs); denom = 1 + z**2/n
    centre = p + z*z/(2*n)
    pm = z*np.sqrt((p*(1-p) + z*z/(4*n))/n)
    lo = (centre - pm) / denom
    hi = (centre + pm) / denom
    return p, lo, hi
