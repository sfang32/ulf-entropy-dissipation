"""tsallis_scan.py -- q- and bin-sensitivity of the Tsallis entropy measure.

Context: Tsallis_q1.5 is the ONLY metric in the battery with a POSITIVE sign and a
strong latitude trend (rho(lat, coupling) = +0.80 on Pc5). The Qwen figure review
flagged it as possibly parameter-driven. Note its definition differs from the others:
it is the non-extensive entropy of the 64-bin AMPLITUDE histogram (not of the
permutation/spectral distribution).

Here we scan q in {0.5,1,1.5,2,2.5,3} and bins in {16,32,64,128} on real 10-min
windows and report rho(Tsallis, log E_band) to decide whether sign and latitude
trend are robust or artefacts of the chosen parameters.

Run on the server: python tsallis_scan.py [--stations MHT NAT SYS] [--max-days N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import run_multiband as mb

RES = Path("/data/ssd-stor/fsf/workspaces/research_ulf/ground_pc5/"
           "v8_entropy_dissipation/results")
QS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BINS = [16, 32, 64, 128]
WIN, EDGE = 600, 1800


def tsallis_multi(x, qs, bins):
    """One histogram -> Tsallis entropy for several q (S_q=(1-sum p^q)/(q-1))."""
    x = x[np.isfinite(x)]
    if x.size < bins:
        return {q: np.nan for q in qs}
    p, _ = np.histogram(x, bins=bins)
    p = p[p > 0].astype(float)
    p /= p.sum()
    out = {}
    for q in qs:
        out[q] = float(-np.sum(p * np.log(p))) if abs(q - 1.0) < 1e-9 \
            else float((1.0 - np.sum(p ** q)) / (q - 1.0))
    return out


def scan_station(station, year, max_days=0):
    days = mb.load_zw(station, year)
    if max_days:
        days = days[:max_days]
    rows = []
    for day, s in days:
        raw = s.to_numpy()
        f5 = mb.band_filter(raw, "Pc5", 1.0)
        fa = mb.band_filter(raw, "all", 1.0)
        for w0 in range(0, raw.size - WIN + 1, WIN):
            if (w0 < EDGE) or (w0 + WIN + EDGE > raw.size):
                continue
            rw = raw[w0:w0 + WIN]
            if np.all(~np.isfinite(rw)):
                continue
            rec = {"station": station, "time": s.index[w0],
                   "E_pc5": float(np.mean(f5[w0:w0 + WIN] ** 2)),
                   "E_all": float(np.mean(fa[w0:w0 + WIN] ** 2))}
            for b in BINS:
                for q, v in tsallis_multi(rw, QS, b).items():
                    rec[f"T_b{b}_q{q}"] = v
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=["MHT", "NAT", "SYS"])
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--max-days", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for st in a.stations:
        df = scan_station(st, a.year, a.max_days)
        print("[scan] %s: %d windows" % (st, len(df)), flush=True)
        for b in BINS:
            for q in QS:
                col = f"T_b{b}_q{q}"
                ok = df[col].notna() & df["E_pc5"].notna()
                if ok.sum() < 100:
                    continue
                sub = df[ok]
                le5 = np.log10(sub["E_pc5"].clip(1e-12))
                lea = np.log10(sub["E_all"].clip(1e-12))
                rows.append({"station": st, "bins": b, "q": q, "n": int(ok.sum()),
                             "rho_Epc5": float(sub[col].corr(le5, method="spearman")),
                             "rho_Eall": float(sub[col].corr(lea, method="spearman"))})
    tab = pd.DataFrame(rows)
    out = RES / "tsallis_scan.csv"
    tab.to_csv(out, index=False)
    print("[save] %s" % out, flush=True)

    for col in ("rho_Epc5", "rho_Eall"):
        piv = tab.pivot_table(index=["bins", "q"], columns="station",
                              values=col, aggfunc="mean")
        piv["MEAN"] = piv.mean(axis=1)
        print("\n=== %s ===" % col)
        print(piv.round(3).to_string(), flush=True)

    # ---- figure: rho vs q, one panel per band-channel ----
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    for ax, col, ttl in zip(axes, ("rho_Epc5", "rho_Eall"),
                            (r"vs $\log E_{Pc5}$", r"vs $\log E_{\rm all}$")):
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axhspan(-0.10, 0.10, color="gray", alpha=0.13, lw=0)
        for b, c in zip(BINS, ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]):
            g = tab[tab.bins == b].groupby("q")[col]
            mu = g.mean()
            se = g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))
            ax.errorbar(mu.index, mu.values, yerr=se.values, fmt="-o",
                        color=c, ms=5, lw=1.5, capsize=2.5, elinewidth=0.9,
                        label="bins=%d" % b)
        ax.set_xlabel("Tsallis $q$")
        ax.set_title(ttl)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_ylim(-0.15, 0.55)
    axes[0].set_ylabel(r"Spearman $\rho$ (mean $\pm$ SEM over stations)")
    for a_ in axes:
        a_.tick_params(labelleft=True)          # label both panels: they share ylim
    # Shared figure legend at the top; both panels use the same bin curves.
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    # In-figure title removed; described in the manuscript caption.
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RES / "tsallis_scan.png", dpi=300)
    print("[save] tsallis_scan.png", flush=True)

    # ---- verdict ----
    for col in ("rho_Epc5", "rho_Eall"):
        d = tab.groupby("q")[col].mean()
        print("\n[verdict %s] sign across all (bins,q): %d/%d negative; "
              "rho range [%+.3f, %+.3f]" % (col, int((tab[col] < 0).sum()),
                                            len(tab), tab[col].min(), tab[col].max()))


if __name__ == "__main__":
    main()
