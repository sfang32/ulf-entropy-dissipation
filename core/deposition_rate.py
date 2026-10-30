"""ULF -> ionospheric Joule heating deposition rate (ground-magnetometer based).

Method (Rae et al. 2007 / 2025 JGR; van Hazendonk et al. 2024 JGR):
  Field-line-resonance (FLR) shielding relates the ground horizontal magnetic
  disturbance dB to the ionospheric horizontal electric field E via
        E = dB / (mu0 * Sigma_p)
  The downward Poynting flux (energy flux into the ionosphere) is
        S = Sigma_p * E^2 = dB^2 / (mu0^2 * Sigma_p)      [W/m^2]
  Integrating S over the footprint area and over time gives the deposited
  energy / energy-deposition rate. We compute the per-window mean Poynting
  flux and the integrated energy per unit area.

NOTE (prototype): the Pedersen conductance Sigma_p uses a simple empirical
placeholder (mid-latitude baseline modulated by Kp). Replace with Robinson 1987
or AMPERE-assimilated Sigma_p before publication. van Hazendonk (2024) ISR
measured dissipation rates are used as the absolute anchor in the final step.
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0 * np.pi * 1e-7  # H/m


def pedersen_conductance(mlat_deg: float, kp: float) -> float:
    """Prototype empirical Pedersen conductance [S].

    Placeholder: constant mid-latitude baseline modulated by Kp.
    TODO(final): Robinson et al. 1987 empirical model, or AMPERE assimilation.
    """
    base = 5.0  # S, typical mid-latitude value
    return float(base * (1.0 + 0.3 * kp))


def poynting_flux(dB_T: np.ndarray, sigma_p: float) -> np.ndarray:
    """Downward Poynting flux S = dB^2 / (mu0^2 * Sigma_p)  [W/m^2].

    dB_T: horizontal magnetic disturbance amplitude [T] (ULF perturbation).
    Vectorized over an array of amplitudes.
    """
    dB_T = np.asarray(dB_T, dtype=float)
    return dB_T**2 / (MU0**2 * sigma_p)


def window_deposition(h_series_nT: np.ndarray, mlat_deg: float, kp: float,
                      fs: float = 1.0):
    """Per-window ULF energy deposition from a 1 Hz H series (nT).

    Returns
    -------
    S_mean : float   mean downward Poynting flux over the window [W/m^2]
    E_int  : float   integrated energy per unit area over the window [J/m^2]
    dB_rms : float   RMS horizontal disturbance amplitude [nT]
    """
    h = np.asarray(h_series_nT, dtype=float)
    mask = ~np.isnan(h)
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan, np.nan
    dh = h[mask] - np.nanmean(h[mask])          # isolate ULF disturbance
    dB_rms_nT = float(np.sqrt(np.nanmean(dh**2)))
    dB_T = dB_rms_nT * 1e-9                       # nT -> T
    sig = pedersen_conductance(mlat_deg, kp)
    S_mean = float(poynting_flux(dB_T, sig))      # W/m^2
    dt = 1.0 / fs
    E_int = S_mean * n * dt                       # J/m^2
    return S_mean, E_int, dB_rms_nT


def series_deposition(h_1hz: np.ndarray, mlat_deg: float, kp_series: np.ndarray,
                      win: int = 1800, step: int = 1800, fs: float = 1.0):
    """Sliding/non-overlapping window deposition rate over a long H series.

    Returns arrays (S_mean[per window], E_int[per window], dB_rms[per window])
    aligned to window start indices.
    """
    h = np.asarray(h_1hz, dtype=float)
    n = h.size
    S_out, E_out, dB_out = [], [], []
    for s in range(0, n - win + 1, step):
        seg = h[s:s + win]
        kp = float(np.nanmean(kp_series[s:s + win])) if kp_series is not None else 0.0
        Sm, Em, dB = window_deposition(seg, mlat_deg, kp, fs=fs)
        S_out.append(Sm); E_out.append(Em); dB_out.append(dB)
    return (np.array(S_out), np.array(E_out), np.array(dB_out))
