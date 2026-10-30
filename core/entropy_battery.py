"""Self-contained entropy / complexity battery for ULF ground H series.

Server-ready: NUMPY ONLY (no ordpy / antropy / infomeasure needed).

Direction H (Entropy special issue) augmented battery, deliberately avoiding
transfer entropy (owned by v7_TE submitted to JASTP). All measures are
per fixed window so they align with deposition_rate.py estimates.

Measures
--------
Baseline (v6):
  * permutation entropy (Bandt-Pompe, normalized)
  * spectral entropy (full + band-limited Pc3/Pc4/Pc5)
  * sample entropy, multiscale entropy
  * Tsallis (histogram) entropy
Augmented (H):
  * weighted permutation entropy  (amplitude-aware -> ties to wave power)
  * complexity-entropy (H,C) plane (Rosso 2007, Shannon)
  * Tsallis & Renyi generalized CH curves (non-extensive / non-equilibrium)
  * SVD entropy (coherence / effective dimension)
  * ordinal irreversibility (forward vs time-reversed pattern divergence)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

BANDS_HZ = {"Pc3": (22e-3, 100e-3), "Pc4": (6.7e-3, 22e-3), "Pc5": (1.7e-3, 6.7e-3)}


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


# --------------------------------------------------------------------------
# Bandt-Pompe ordinal pattern distribution
# --------------------------------------------------------------------------
def _pattern_distribution(x, m, tau=1):
    x = _clean(x)
    n = x.size
    nvec = n - (m - 1) * tau
    if nvec <= 0:
        return None
    idx = np.arange(nvec)[:, None] + np.arange(m)[None, :] * tau
    pats = x[idx]
    order = np.argsort(pats, axis=1, kind="stable")
    uniq, counts = np.unique(order, axis=0, return_counts=True)
    return counts / counts.sum()


def permutation_entropy(x, m=4, tau=1):
    p = _pattern_distribution(x, m, tau)
    if p is None:
        return np.nan
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(p.size))  # normalized by log(m!)


def weighted_permutation_entropy(x, m=4, tau=1):
    """Amplitude-weighted permutation entropy (Fadlallah et al. 2013),
    normalized by log(#distinct patterns) -> [0, 1] like PE."""
    x = _clean(x)
    n = x.size
    nvec = n - (m - 1) * tau
    if nvec <= 0:
        return np.nan
    idx = np.arange(nvec)[:, None] + np.arange(m)[None, :] * tau
    pats = x[idx]
    order = np.argsort(pats, axis=1, kind="stable")
    # weight per pattern = variance of the embedded window
    w = np.sum((pats - pats.mean(axis=1, keepdims=True)) ** 2, axis=1) / (m - 1)
    uniq, inv = np.unique(order, axis=0, return_inverse=True)
    wsum = np.bincount(inv, weights=w)
    wsum = wsum[wsum > 0]
    p = wsum / wsum.sum()
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(uniq.shape[0]))


# --------------------------------------------------------------------------
# Spectral entropy (Welch-free, Hann-windowed periodogram)
# --------------------------------------------------------------------------
def spectral_entropy(x, fs=1.0, f1=None, f2=None, normalize=True):
    x = _clean(x)
    n = x.size
    if n < 8:
        return np.nan
    x = x - np.mean(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    Pxx = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    if f1 is not None:
        band = (f >= f1) & (f <= f2)
        Pxx = Pxx[band]
    s = Pxx.sum()
    if s <= 0:
        return np.nan
    p = Pxx / s
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    if normalize:
        H = H / math.log(p.size)
    return float(H)


# --------------------------------------------------------------------------
# Sample & multiscale entropy (Richman & Moorman 2000)
# --------------------------------------------------------------------------
def _phi(x, m, r):
    x = _clean(x)
    N = x.size
    if N < m + 1:
        return 0.0
    templates = np.lib.stride_tricks.sliding_window_view(x, m)
    tree = cKDTree(templates)
    # all pairs within r (chebyshev), minus self-matches (i == j)
    cnt = tree.count_neighbors(tree, r) - (N - m + 1)
    return cnt / (N - m + 1)


def sample_entropy(x, m=2, r=None):
    x = _clean(x)
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0 or x.size < m + 2:
        return np.nan
    A = _phi(x, m + 1, r)
    B = _phi(x, m, r)
    if B <= 0 or A <= 0:      # A=0: too regular for m+1 templates -> undefined
        return np.nan
    return float(-math.log(A / B))


def multiscale_entropy(x, m=2, r=None, scales=7):
    x = _clean(x)
    if r is None:
        r = 0.2 * np.std(x)
    out = []
    for s in range(1, scales + 1):
        if x.size < s * 2:
            break
        coarse = x[:s * (x.size // s)].reshape(-1, s).mean(axis=1)
        out.append(sample_entropy(coarse, m=m, r=r))
    return np.array(out) if out else np.array([np.nan])


# --------------------------------------------------------------------------
# Tsallis (histogram) entropy
# --------------------------------------------------------------------------
def tsallis_entropy(x, q=1.5, bins=64):
    x = _clean(x)
    if x.size < bins:
        return np.nan
    p, _ = np.histogram(x, bins=bins, density=False)
    p = p[p > 0].astype(float)
    p = p / p.sum()
    if abs(q - 1.0) < 1e-9:
        return float(-np.sum(p * np.log(p)))
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


# --------------------------------------------------------------------------
# Complexity-entropy (H, C) plane (Rosso et al. 2007)
# --------------------------------------------------------------------------
def _js_divergence(p, pe):
    """Standard Jensen-Shannon divergence (nats), symmetric, bounded by ln(2)."""
    p = p[p > 0]; pe = pe[pe > 0]
    m = 0.5 * (p + pe)
    m = m[m > 0]
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(pe, m)


def complexity_entropy(x, m=4, tau=1):
    p = _pattern_distribution(x, m, tau)
    if p is None:
        return (np.nan, np.nan)
    p = p[p > 0]
    N = p.size
    if N <= 1:
        return (0.0, 0.0)
    H = -np.sum(p * np.log(p)) / math.log(N)
    pe = np.ones(N) / N
    Q = _js_divergence(p, pe)
    C = H * (Q / math.log(2))  # JS bounded by ln(2)
    return (float(H), float(C))


# --------------------------------------------------------------------------
# Tsallis / Renyi generalized CH curves (non-extensive thermodynamics)
# --------------------------------------------------------------------------
def _tsallis_entropy_dist(p, q):
    p = p[p > 0]
    if abs(q - 1.0) < 1e-9:
        return -np.sum(p * np.log(p))
    return (1.0 - np.sum(p ** q)) / (q - 1.0)


def _renyi_entropy_dist(p, alpha):
    p = p[p > 0]
    if abs(alpha - 1.0) < 1e-9:
        return -np.sum(p * np.log(p))
    return math.log(np.sum(p ** alpha)) / (1.0 - alpha)


def _jensen_tsallis(p, pe, q):
    """Jensen-Tsallis divergence (Nielsen 2010) for two distributions."""
    m = 0.5 * (p + pe)
    m = m[m > 0]; p = p[p > 0]; pe = pe[pe > 0]
    Dp = _tsallis_entropy_dist(m, q) - _tsallis_entropy_dist(p, q)
    Dq = _tsallis_entropy_dist(m, q) - _tsallis_entropy_dist(pe, q)
    return float(0.5 * (Dp + Dq))


def _jensen_renyi(p, pe, alpha):
    m = 0.5 * (p + pe)
    m = m[m > 0]; p = p[p > 0]; pe = pe[pe > 0]
    Dp = _renyi_entropy_dist(m, alpha) - _renyi_entropy_dist(p, alpha)
    Dq = _renyi_entropy_dist(m, alpha) - _renyi_entropy_dist(pe, alpha)
    return float(0.5 * (Dp + Dq))


def tsallis_complexity_entropy(x, q_list, m=4, tau=1):
    """Return (n_q, 2) array of [H_q, C_q] per q (normalized)."""
    p = _pattern_distribution(x, m, tau)
    if p is None:
        return np.full((len(q_list), 2), np.nan)
    p = p[p > 0]
    N = p.size
    if N <= 1:
        return np.full((len(q_list), 2), np.nan)
    pe = np.ones(N) / N
    out = []
    for q in q_list:
        Spat = _tsallis_entropy_dist(p, q)
        Hq = Spat / math.log(N) if q != 1.0 else -np.sum(p * np.log(p)) / math.log(N)
        Q = _jensen_tsallis(p, pe, q)
        Cq = Hq * (Q / math.log(2))  # JS bounded by ln(2)
        out.append([float(Hq), float(Cq)])
    return np.array(out)


def renyi_complexity_entropy(x, alpha_list, m=4, tau=1):
    """Return (n_a, 2) array of [H_a, C_a] per alpha (normalized)."""
    p = _pattern_distribution(x, m, tau)
    if p is None:
        return np.full((len(alpha_list), 2), np.nan)
    p = p[p > 0]
    N = p.size
    if N <= 1:
        return np.full((len(alpha_list), 2), np.nan)
    pe = np.ones(N) / N
    out = []
    for a in alpha_list:
        Hr = _renyi_entropy_dist(p, a) / math.log(N)
        Q = _jensen_renyi(p, pe, a)
        Ca = Hr * (Q / math.log(2))  # JS bounded by ln(2)
        out.append([float(Hr), float(Ca)])
    return np.array(out)


# --------------------------------------------------------------------------
# SVD entropy (coherence / effective dimension)
# --------------------------------------------------------------------------
def svd_entropy(x, m=4, tau=1, normalize=True):
    x = _clean(x)
    n = x.size
    nvec = n - (m - 1) * tau
    if nvec <= 0:
        return np.nan
    idx = np.arange(nvec)[:, None] + np.arange(m)[None, :] * tau
    Y = x[idx]
    S = np.linalg.svd(Y, compute_uv=False)
    s = S[S > 0]
    if s.size == 0:
        return np.nan
    p = s / s.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    if normalize:
        H = H / math.log(m)
    return float(H)


# --------------------------------------------------------------------------
# Ordinal irreversibility (Martinez et al. 2008)
# --------------------------------------------------------------------------
def irreversibility_ordinal(x, m=4, tau=1):
    """Jensen-Shannon divergence between forward and time-reversed ordinal
    pattern distributions. > 0 => preferred time arrow (dissipative/driven)."""
    x = _clean(x)
    n = x.size
    nvec = n - (m - 1) * tau
    if nvec <= 0:
        return np.nan
    idx = np.arange(nvec)[:, None] + np.arange(m)[None, :] * tau
    pats = x[idx]
    fwd = np.argsort(pats, axis=1, kind="stable")
    rev = fwd[:, ::-1]
    uf, cf = np.unique(fwd, axis=0, return_counts=True)
    ur, cr = np.unique(rev, axis=0, return_counts=True)
    pf = cf / cf.sum()
    pr = cr / cr.sum()
    m = 0.5 * (pf + pr)
    good = (m > 0) & (pf > 0) & (pr > 0)
    pf_g, pr_g, m_g = pf[good], pr[good], m[good]
    if pf_g.size == 0:
        return 0.0
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    js = 0.5 * kl(pf_g, m_g) + 0.5 * kl(pr_g, m_g)
    return float(js)


# --------------------------------------------------------------------------
# Full per-window battery
# --------------------------------------------------------------------------
def entropy_battery(h, fs=1.0, m_pe=4, scales=7,
                    q_list=(0.5, 1.0, 1.5, 2.0, 2.5),
                    alpha_list=(0.5, 1.0, 1.5, 2.0, 2.5)):
    h = _clean(h)
    res = {}
    res["SE_full"] = spectral_entropy(h, fs=fs)
    for b, (f1, f2) in BANDS_HZ.items():
        res[f"SE_{b}"] = spectral_entropy(h, fs=fs, f1=f1, f2=f2)
    res["PE"] = permutation_entropy(h, m=m_pe, tau=1)
    res["WPE"] = weighted_permutation_entropy(h, m=m_pe, tau=1)
    sd = np.std(h)
    res["SampEn"] = sample_entropy(h, m=2, r=0.2 * sd) if sd > 0 else np.nan
    res["MSE"] = multiscale_entropy(h, m=2, r=0.2 * sd, scales=scales)
    res["Tsallis_q1.5"] = tsallis_entropy(h, q=1.5)
    Hc, Cc = complexity_entropy(h, m=m_pe, tau=1)
    res["CH_H"], res["CH_C"] = Hc, Cc
    ch_t = tsallis_complexity_entropy(h, q_list, m=m_pe, tau=1)
    for qq, (hh, cc) in zip(q_list, ch_t):
        res[f"CH_Tsallis_q{qq}"] = (float(hh), float(cc))
    ch_r = renyi_complexity_entropy(h, alpha_list, m=m_pe, tau=1)
    for aa, (hh, cc) in zip(alpha_list, ch_r):
        res[f"CH_Renyi_a{aa}"] = (float(hh), float(cc))
    res["SVD"] = svd_entropy(h, m=m_pe, tau=1)
    res["IR_ordinal"] = irreversibility_ordinal(h, m=m_pe, tau=1)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(1800)
    x = np.sin(2 * np.pi * t / 300.0) + 0.3 * rng.standard_normal(1800)
    out = entropy_battery(x)
    for k, v in out.items():
        print(f"{k}: {v}")
