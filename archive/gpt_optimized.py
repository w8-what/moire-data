import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from decimal import Decimal
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import BayesianRidge

OUT = Path("output/gpt_optimized")
IN  = Path("source_data")
OUT.mkdir(parents=True, exist_ok=True)

FIELDS        = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD  = 20.0          # Ω·cm  — resistivities below this → superconducting
SC_DROP_FRAC  = 0.50          # ≥50 % drop within SC_DROP_WINDOW_K → onset transition
SC_DROP_WINDOW_K = 5.0        # K     — window over which we look for a sharp drop
MIN_TEMP_RANGE   = 3.0        # K     — minimum span a phase segment must cover

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size"  : 9,
    "axes.linewidth"   : 1.0,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
})

# ─────────────────────────────────────────────────────────────
# Phase display metadata
# ─────────────────────────────────────────────────────────────
PHASE_META: Dict[str, dict] = {
    "superconducting"  : {"label": "SC",              "color": "#4dbbff"},
    "sc_transition"    : {"label": "SC Transition",   "color": "#a78bfa"},
    "strange_metal"    : {"label": "Strange Metal",   "color": "#f97316"},
    "fermi_liquid"     : {"label": "Fermi Liquid",    "color": "#22c55e"},
    "sublinear_metal"  : {"label": "Sublinear Metal", "color": "#eab308"},
    "insulator"        : {"label": "Insulator",       "color": "#ef4444"},
    "afm_metal"        : {"label": "AFM Metal",       "color": "#06b6d4"},
    "afm_insulator"    : {"label": "AFM Insulator",   "color": "#dc2626"},
    "constant"         : {"label": "Constant",        "color": "#94a3b8"},
}

# Threshold for kink-strength ratio to qualify as an AFM boundary.
# Dimensionless: |Δ slope| / mean(|slope|) across the breakpoint.
AFM_KINK_THRESHOLD = 1.5

# ─────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────

def load_field(E):
    df  = pd.read_csv(IN / f"Rxx_matrix_E-{E}mV_nm.csv")
    T   = df.iloc[:, 0].astype(float).to_numpy()
    nu  = np.array([float(c) for c in df.columns[1:]])
    R   = df.iloc[:, 1:].astype(float).to_numpy()
    return T, nu, R

# ─────────────────────────────────────────────────────────────
# 2.  Superconducting pre-pass
#     Returns a boolean mask (True = SC) and T_c (or None)
# ─────────────────────────────────────────────────────────────

def detect_superconducting(T: np.ndarray, rho: np.ndarray
                           ) -> Tuple[np.ndarray, Optional[float]]:
    """
    Returns
    -------
    sc_mask : bool array, True where the data is in the SC / transition region
    T_c     : estimated onset temperature (None if no SC region found)
    """
    T   = np.asarray(T,   dtype=float)
    rho = np.asarray(rho, dtype=float)

    sc_mask = np.zeros(len(T), dtype=bool)
    T_c: Optional[float] = None

    # --- Hard SC region: ρ < SC_THRESHOLD ----------------------------------
    below = rho < SC_THRESHOLD
    sc_mask |= below

    # --- Transition: sharp drop *towards* the SC threshold -----------------
    # Smooth ρ slightly to suppress noise before computing drop
    rho_sm = uniform_filter1d(rho, size=max(3, len(T) // 40))

    # For each point, look ahead (higher T → lower T) for a drop
    # We scan in *decreasing* T order so we catch the onset from above
    order = np.argsort(T)
    T_s, rho_s = T[order], rho_sm[order]   # sorted low→high T

    # Window in index space that spans ≈ SC_DROP_WINDOW_K
    dT_mean = np.mean(np.diff(T_s)) if len(T_s) > 1 else 1.0
    win     = max(2, int(SC_DROP_WINDOW_K / dT_mean))

    transition_indices = set()
    for k in range(len(T_s) - win):
        rho_hi = rho_s[k + win]   # higher T end of window
        rho_lo = rho_s[k]         # lower  T end of window
        if rho_hi > 0 and (rho_hi - rho_lo) / rho_hi >= SC_DROP_FRAC:
            # Significant drop toward low T — mark entire window as transition
            transition_indices.update(range(k, k + win + 1))

    # Re-map sorted indices back to original
    for k in transition_indices:
        orig_idx = order[k]
        sc_mask[orig_idx] = True

    # Estimate T_c as the highest T where rho_smooth first crosses SC_THRESHOLD
    # (i.e. the onset of the transition from above)
    if sc_mask.any():
        # Walk down from the highest marked temperature
        marked_T = T[sc_mask]
        T_c = float(np.max(marked_T))

        # Refine: find the steepest descent point within the marked window
        marked_sorted = np.sort(marked_T)
        if len(marked_sorted) > 2:
            # Indices in sorted array
            lo_idx = np.searchsorted(T_s, marked_sorted[0])
            hi_idx = np.searchsorted(T_s, marked_sorted[-1])
            seg_rho = rho_s[lo_idx : hi_idx + 1]
            seg_T   = T_s  [lo_idx : hi_idx + 1]
            if len(seg_rho) > 1:
                drho = np.diff(seg_rho) / np.diff(seg_T)
                peak = np.argmin(drho)      # most negative slope
                T_c  = float(seg_T[peak])

    return sc_mask, T_c

# ─────────────────────────────────────────────────────────────
# 3.  Model definitions
#     Constraints enforced at prediction time:
#       • quadratic only accepted when leading coef > 0 (concave up)
#       • sublinear (log) only accepted when slope > 0
# ─────────────────────────────────────────────────────────────

def raw_features_constant(T):
    return np.empty((len(T), 0))

def raw_features_linear(T):
    return T.reshape(-1, 1)

def raw_features_quadratic(T):
    # [T², T]  — Fermi-liquid form ρ = aT² + bT + c, a > 0 enforced later
    return np.column_stack([T**2, T])

def raw_features_log(T):
    # Sublinear: ρ = a·log(T) + c, a > 0 enforced later
    return np.log(T).reshape(-1, 1)

@dataclass(frozen=True)
class ModelSpec:
    name             : str
    feature_fn       : Callable[[np.ndarray], np.ndarray]
    needs_positive_T : bool = False
    physics_label    : str  = "constant"

MODELS = [
    ModelSpec("constant",      raw_features_constant,  False, "constant"),
    ModelSpec("linear",        raw_features_linear,    False, "strange_metal"),
    ModelSpec("quadratic",     raw_features_quadratic, False, "fermi_liquid"),
    ModelSpec("sublinear_log", raw_features_log,       True,  "sublinear_metal"),
]
MODEL_LOOKUP = {m.name: m for m in MODELS}

@dataclass
class SegmentFit:
    i                    : int
    j                    : int
    model_name           : str
    physics_label        : str
    beta                 : np.ndarray
    feature_mean         : np.ndarray
    feature_scale        : np.ndarray
    log_marginal_likelihood: float
    model_obj            : BayesianRidge

    @property
    def length(self):
        return self.j - self.i

def standardize_features(F):
    if F.shape[1] == 0:
        return F, np.array([]), np.array([])
    mu    = F.mean(axis=0)
    scale = F.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (F - mu) / scale, mu, scale

def predict_segment(fit: SegmentFit, T_new, return_std=False):
    T_new  = np.asarray(T_new, dtype=float)
    model  = MODEL_LOOKUP[fit.model_name]
    F      = model.feature_fn(T_new)
    if F.shape[1] > 0:
        F = (F - fit.feature_mean) / fit.feature_scale
    if return_std:
        return fit.model_obj.predict(F, return_std=True)
    return fit.model_obj.predict(F)

def _slope_is_positive(fit: SegmentFit, T_seg: np.ndarray) -> bool:
    """Estimate dρ/dT over the segment and return True if net positive."""
    if len(T_seg) < 2:
        return True
    y = predict_segment(fit, T_seg)
    return float(y[-1] - y[0]) > 0

def _quadratic_a_positive(fit: SegmentFit) -> bool:
    """Return True iff the quadratic leading coefficient is positive (concave up)."""
    if fit.model_name != "quadratic":
        return True
    # Coefficient vector is in standardised space.  Un-standardise T² term.
    # beta[0] corresponds to T²; sign is preserved through positive scaling.
    return float(fit.beta[0]) > 0

# ─────────────────────────────────────────────────────────────
# 4.  Physics label assignment
#     Called *after* BayesianRidge fit to enforce physical constraints
#     and assign the correct phase label.
# ─────────────────────────────────────────────────────────────

def assign_physics_label(fit: SegmentFit, T_seg: np.ndarray) -> str:
    """
    Map a model fit onto a physics label, enforcing slope/curvature rules.
    Returns the physics label string.
    """
    name     = fit.model_name
    pos_slope = _slope_is_positive(fit, T_seg)

    if name == "constant":
        return "constant"

    if name == "linear":
        return "strange_metal" if pos_slope else "insulator"

    if name == "quadratic":
        if not _quadratic_a_positive(fit):
            # Concave-down quadratic is not physical — treat as insulator or
            # fall back to linear slope sign
            return "insulator" if not pos_slope else "strange_metal"
        return "fermi_liquid" if pos_slope else "insulator"

    if name == "sublinear_log":
        # Only valid as sublinear metal when slope is positive
        if pos_slope:
            return "sublinear_metal"
        else:
            return "insulator"

    return name   # fallback

# ─────────────────────────────────────────────────────────────
# 5.  Bayesian fitting (single interval, single model)
# ─────────────────────────────────────────────────────────────

def fit_single_model_on_interval(
    T, rho, i, j, model: ModelSpec
) -> Optional[SegmentFit]:
    x = np.asarray(T[i:j],   dtype=float)
    y = np.asarray(rho[i:j], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    n     = len(x)

    if n < 3 or (model.needs_positive_T and np.any(x <= 0)):
        return None

    try:
        F_raw = model.feature_fn(x)
    except Exception:
        return None

    if not np.all(np.isfinite(F_raw)):
        return None

    F, feature_mean, feature_scale = standardize_features(F_raw)
    if n <= F.shape[1] + 1:
        return None

    br = BayesianRidge(compute_score=True)
    try:
        br.fit(F, y)
    except Exception:
        return None

    # Determine physics label *before* filtering
    proto = SegmentFit(
        i=i, j=j,
        model_name=model.name,
        physics_label=model.physics_label,
        beta=br.coef_,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        log_marginal_likelihood=-float(br.scores_[-1]),
        model_obj=br,
    )
    phys_label = assign_physics_label(proto, x)

    # ── Physical constraint filtering ──────────────────────────
    # Reject concave-down quadratics — do NOT fall back, let DP choose another model
    if model.name == "quadratic" and not _quadratic_a_positive(proto):
        return None
    # Reject log fits with negative slope
    if model.name == "sublinear_log" and not _slope_is_positive(proto, x):
        return None

    return SegmentFit(
        i=i, j=j,
        model_name=model.name,
        physics_label=phys_label,
        beta=br.coef_,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        log_marginal_likelihood=-float(br.scores_[-1]),
        model_obj=br,
    )

# ─────────────────────────────────────────────────────────────
# 6.  Model-aware DP with temperature-range constraint
# ─────────────────────────────────────────────────────────────

def get_physics_label_at_T(
    segments: List[SegmentFit], T_val: float, T_array: np.ndarray
) -> Optional[str]:
    for seg in segments:
        lo = T_array[seg.i]
        hi = T_array[min(seg.j, len(T_array) - 1)]
        if lo <= T_val <= hi:
            return seg.physics_label
    return None

def segment_linecut_dp_constrained(
    T, rho,
    min_temp_range   = MIN_TEMP_RANGE,
    max_window       = None,
    breakpoint_penalty = 5.0,
    jump_weight        = 15.0,
    neighbor_segments  = None,
    neighbor_weight    = 3.0,
    models             = MODELS,
) -> Tuple[np.ndarray, np.ndarray, List[SegmentFit], float]:
    """
    Segmentation DP.

    Key change vs original:
      min_window (index count) → min_temp_range (Kelvin), so that segment
      length is enforced in physical units rather than sample-count units.
    """
    T   = np.asarray(T,   dtype=float)
    rho = np.asarray(rho, dtype=float)
    order = np.argsort(T)
    T, rho = T[order], rho[order]
    N = len(T)

    if max_window is None:
        max_window = N

    var_rho = float(np.var(rho)) if np.var(rho) > 1e-12 else 1.0

    # Convert min_temp_range to a minimum index gap for each starting index
    # (grid may be non-uniform so we compute per-point)
    def min_j(i: int) -> int:
        """Smallest j such that T[j-1] - T[i] >= min_temp_range."""
        target = T[i] + min_temp_range
        idx    = np.searchsorted(T, target, side="right")
        return max(i + 2, int(idx))  # at least 2 points

    # ── 1. Precompute all interval fits ───────────────────────
    interval_fit  = [[[ None for _ in models ] for _ in range(N + 1)] for _ in range(N)]
    interval_cost = np.full((N, N + 1, len(models)), np.inf)

    for i in range(N):
        j_min = min_j(i)
        j_max = min(N, i + max_window) + 1
        for j in range(j_min, j_max):
            for m_idx, model in enumerate(models):
                fit = fit_single_model_on_interval(T, rho, i, j, model)
                if fit is not None:
                    interval_cost[i, j, m_idx] = fit.log_marginal_likelihood
                    interval_fit[i][j][m_idx]  = fit

    # ── 2. DP arrays ──────────────────────────────────────────
    dp     = np.full((N + 1, len(models)), np.inf)
    prev_i = np.full((N + 1, len(models)), -1, dtype=int)
    prev_m = np.full((N + 1, len(models)), -1, dtype=int)

    for m in range(len(models)):
        dp[0, m] = 0.0

    # ── 3. DP transitions ──────────────────────────────────────
    for j in range(2, N + 1):
        for m_idx, model in enumerate(models):
            for i in range(max(0, j - max_window), j):
                if T[min(j, N) - 1] - T[i] < min_temp_range and i > 0:
                    continue   # segment too short in temperature
                fit_curr = interval_fit[i][j][m_idx]
                if fit_curr is None:
                    continue

                # Neighbor (2-D phase) prior
                prior_cost = 0.0
                if neighbor_segments is not None:
                    T_mid      = T[(i + j) // 2]
                    neigh_lbl  = get_physics_label_at_T(neighbor_segments, T_mid, T)
                    if neigh_lbl and neigh_lbl != fit_curr.physics_label:
                        prior_cost = neighbor_weight

                if i == 0:
                    cost = interval_cost[i, j, m_idx] + prior_cost
                    if cost < dp[j, m_idx]:
                        dp[j, m_idx]     = cost
                        prev_i[j, m_idx] = 0
                else:
                    for p_idx in range(len(models)):
                        if not np.isfinite(dp[i, p_idx]):
                            continue
                        start_prev = prev_i[i, p_idx]
                        if start_prev == -1:
                            continue
                        fit_prev = interval_fit[start_prev][i][p_idx]
                        if fit_prev is None:
                            continue

                        # C0 continuity (jump) penalty
                        y_prev = predict_segment(fit_prev, [T[i]])[0]
                        y_curr = predict_segment(fit_curr, [T[i]])[0]
                        jump_cost = jump_weight * ((y_prev - y_curr) ** 2) / var_rho

                        cost = (
                            dp[i, p_idx]
                            + interval_cost[i, j, m_idx]
                            + breakpoint_penalty
                            + jump_cost
                            + prior_cost
                        )
                        if cost < dp[j, m_idx]:
                            dp[j, m_idx]     = cost
                            prev_i[j, m_idx] = i
                            prev_m[j, m_idx] = p_idx

    # ── 4. Reconstruct path ───────────────────────────────────
    best_m = int(np.argmin(dp[N, :]))
    if not np.isfinite(dp[N, best_m]):
        raise RuntimeError("No valid segmentation found.")

    segments: List[SegmentFit] = []
    j, curr_m = N, best_m
    while j > 0:
        i = prev_i[j, curr_m]
        segments.append(interval_fit[i][j][curr_m])
        curr_m, j = prev_m[j, curr_m], i

    segments.reverse()
    return T, rho, segments, float(dp[N, best_m])

# ─────────────────────────────────────────────────────────────
# 7.  AFM post-hoc relabelling
#
#  Run *after* DP segmentation on the full segment list.
#  For each internal breakpoint, we measure the "kink strength":
#
#      kink = |slope_above - slope_below| / mean(|slope_above|, |slope_below|)
#
#  where slope_X is the mean dρ/dT of the segment on side X of the
#  breakpoint, evaluated using the Bayesian fit (not raw finite differences).
#
#  If kink > AFM_KINK_THRESHOLD the breakpoint is classified as a Néel
#  transition (T_N).  The segment *below* T_N is then relabelled:
#    • fermi_liquid / strange_metal / sublinear_metal  →  afm_metal
#    • insulator                                        →  afm_insulator
#
#  The segment above T_N keeps its original label — above T_N the material
#  is in its paramagnetic phase and the existing labels are correct.
#
#  We never relabel the *topmost* segment (it has no "above" neighbour to
#  create a kink) and we propagate relabelling downward: if seg[k] is already
#  afm_metal/insulator, seg[k-1] (further below) is also a candidate.
# ─────────────────────────────────────────────────────────────

def _mean_slope(fit: SegmentFit, T_seg: np.ndarray) -> float:
    """Mean dρ/dT over a segment, using the Bayesian fit."""
    if len(T_seg) < 2:
        return 0.0
    y = predict_segment(fit, T_seg)
    # Central-difference average
    dT  = T_seg[-1] - T_seg[0]
    return float(y[-1] - y[0]) / dT if abs(dT) > 1e-12 else 0.0


def relabel_afm_segments(
    segments : List[SegmentFit],
    T        : np.ndarray,
    kink_threshold: float = AFM_KINK_THRESHOLD,
) -> Tuple[List[SegmentFit], List[int]]:
    """
    Parameters
    ----------
    segments       : ordered list of SegmentFit from the DP
    T              : temperature array (same indexing as segments)
    kink_threshold : AFM_KINK_THRESHOLD

    Returns
    -------
    relabelled_segments : new list with physics_label updated where appropriate
    neel_indices        : list of segment indices *above* which a T_N was detected
                          (i.e. the breakpoint between seg[k] and seg[k+1])
    """
    if len(segments) < 2:
        return list(segments), []

    # Build mutable copy (dataclass is not frozen here so we can reassign label)
    segs = list(segments)

    # Slopes for each segment (evaluated over its own T range)
    slopes = []
    for seg in segs:
        T_seg  = T[seg.i : seg.j]
        slopes.append(_mean_slope(seg, T_seg))

    # Identify Néel breakpoints — scan from top downward
    # (high index = high T = paramagnetic side)
    neel_indices: List[int] = []   # index k means breakpoint between segs[k] and segs[k+1]

    for k in range(len(segs) - 1):
        s_lo = slopes[k]       # slope of segment below breakpoint
        s_hi = slopes[k + 1]   # slope of segment above breakpoint

        denom = (abs(s_lo) + abs(s_hi)) / 2.0
        if denom < 1e-12:
            continue

        kink = abs(s_hi - s_lo) / denom

        if kink >= kink_threshold:
            neel_indices.append(k)

    # Relabel downward from each Néel point
    # Once a segment is below *any* Néel point it becomes AFM
    if not neel_indices:
        return segs, []

    lowest_neel = min(neel_indices)   # everything at index ≤ lowest_neel is AFM

    _METALLIC = {"fermi_liquid", "strange_metal", "sublinear_metal", "constant"}
    _INSULATING = {"insulator"}

    for k in range(lowest_neel + 1):   # k = 0 … lowest_neel (inclusive below breakpoint)
        seg = segs[k]
        old = seg.physics_label
        if old in _METALLIC:
            new_label = "afm_metal"
        elif old in _INSULATING:
            new_label = "afm_insulator"
        else:
            continue   # SC or already AFM — don't touch

        # SegmentFit is a regular dataclass (not frozen), so direct assignment works
        seg.physics_label = new_label

    return segs, neel_indices


# ─────────────────────────────────────────────────────────────
# 8.  Smooth composite curve
# ─────────────────────────────────────────────────────────────

def generate_smooth_curve(T_full, segments: List[SegmentFit]) -> np.ndarray:  # noqa: E302
    if not segments:
        return np.zeros_like(T_full)
    preds = [predict_segment(s, T_full) for s in segments]
    if len(segments) == 1:
        return preds[0]

    composite = np.copy(preds[0])
    delta_T   = (T_full[-1] - T_full[0]) * 0.05
    for idx in range(len(segments) - 1):
        T_c     = T_full[segments[idx].j] if segments[idx].j < len(T_full) else T_full[-1]
        weights = 1.0 / (1.0 + np.exp(-(T_full - T_c) / delta_T))
        composite = (1 - weights) * composite + weights * preds[idx + 1]
    return composite

# ─────────────────────────────────────────────────────────────
# 9.  Plotting
# ─────────────────────────────────────────────────────────────

def plot_behavior_fits(
    nu          : float,
    T           : np.ndarray,
    rho         : np.ndarray,
    segments    : List[SegmentFit],
    sc_mask     : np.ndarray,
    T_c         : Optional[float],
    neel_indices: Optional[List[int]] = None,
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(f"Phase Extraction  |  filling ν = {nu:.3f}", fontsize=10, fontweight="bold")

    # ── Left panel: raw data with SC region highlighted ──────
    ax = axes[0]
    ax.set(ylabel="Resistivity (Ω·cm)", xlabel="Temperature (K)", title="Raw Data")
    ax.axhline(SC_THRESHOLD, color="#4dbbff", lw=1.0, ls="--", alpha=0.7,
               label=f"SC threshold ({SC_THRESHOLD} Ω·cm)")

    if sc_mask.any():
        T_sc  = T[sc_mask]
        ax.axvspan(T_sc.min(), T_sc.max(), alpha=0.12, color="#4dbbff", label="SC / transition region")

    if T_c is not None:
        ax.axvline(T_c, color="#a78bfa", lw=1.2, ls=":", label=f"$T_c$ ≈ {T_c:.1f} K")

    ax.plot(T[~sc_mask], rho[~sc_mask], "ko",
            markerfacecolor="none", markeredgecolor="black", ms=3.5, linestyle="none")
    ax.plot(T[sc_mask],  rho[sc_mask],  "o",
            color="#4dbbff", ms=3.5, linestyle="none", alpha=0.7)
    ax.legend(fontsize=7, framealpha=0.7)

    # ── Right panel: phase fits ───────────────────────────────
    ax = axes[1]
    ax.set(ylabel="Resistivity (Ω·cm)", xlabel="Temperature (K)",
           title="Bayesian Phase Segmentation")
    ax.axhline(SC_THRESHOLD, color="#4dbbff", lw=1.0, ls="--", alpha=0.5)

    # SC region shading (no fit drawn — unphysical)
    if sc_mask.any():
        T_sc = T[sc_mask]
        ax.axvspan(T_sc.min(), T_sc.max(), alpha=0.15, color="#4dbbff")
        ax.text(
            T_sc.mean(), SC_THRESHOLD * 0.5,
            "SC", ha="center", va="center",
            fontsize=8, color="#1a85cc", fontweight="bold",
        )

    if T_c is not None:
        ax.axvline(T_c, color="#a78bfa", lw=1.4, ls=":",
                   label=f"$T_c$ ≈ {T_c:.1f} K", zorder=6)

    # T_N markers: one dashed line per Néel breakpoint
    if neel_indices:
        neel_labeled = False
        for k in neel_indices:
            if k < len(segments):
                seg      = segments[k]
                t_n      = T[seg.j] if seg.j < len(T) else T[-1]
                lbl      = f"$T_N$ ≈ {t_n:.1f} K" if not neel_labeled else None
                ax.axvline(t_n, color="#06b6d4", lw=1.3, ls="-.",
                           label=lbl, zorder=6, alpha=0.85)
                ax.text(t_n, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else rho.max(),
                        f" $T_N$={t_n:.1f}K", fontsize=6.5, color="#06b6d4",
                        va="top", rotation=90, clip_on=True)
                neel_labeled = True

    # Background: data points (non-SC in grey, SC in blue)
    ax.plot(T[~sc_mask], rho[~sc_mask], "ko",
            markerfacecolor="none", markeredgecolor="grey",
            ms=3.5, linestyle="none", alpha=0.45, zorder=1)
    ax.plot(T[sc_mask],  rho[sc_mask],  "o",
            color="#4dbbff", ms=3.5, linestyle="none", alpha=0.5, zorder=1)

    plotted_labels: set = set()

    for seg in segments:
        meta  = PHASE_META.get(seg.physics_label, {"label": seg.physics_label, "color": "#888888"})
        color = meta["color"]
        label = meta["label"] if seg.physics_label not in plotted_labels else None
        if label:
            plotted_labels.add(seg.physics_label)

        T_seg             = T[seg.i : seg.j]
        y_mean, y_std     = predict_segment(seg, T_seg, return_std=True)

        ax.fill_between(T_seg, y_mean - 2 * y_std, y_mean + 2 * y_std,
                        color=color, alpha=0.18, zorder=2)
        ax.plot(T_seg, y_mean, color=color, label=label, linewidth=2.2, zorder=3)

    # Smooth composite envelope (non-SC only)
    non_sc_segs = [s for s in segments if s.physics_label not in ("superconducting", "sc_transition")]
    if non_sc_segs:
        T_smooth   = np.linspace(T[~sc_mask].min(), T[~sc_mask].max(), 400)
        rho_smooth = generate_smooth_curve(T_smooth, non_sc_segs)
        ax.plot(T_smooth, rho_smooth, "k--", alpha=0.6, lw=1.4,
                zorder=4, label="Smooth envelope")

    # Breakpoint markers
    for seg in segments[:-1]:
        t_bp  = T[seg.j] if seg.j < len(T) else T[-1]
        r_bp  = rho[np.argmin(np.abs(T - t_bp))]
        ax.plot(t_bp, r_bp, "ro", ms=5.5, zorder=7)

    ax.legend(fontsize=7.5, framealpha=0.8)

    path = OUT / f"fit_filling_{Decimal(nu).quantize(Decimal('0.000'))}.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────
# 10.  Global phase extraction pipeline
# ─────────────────────────────────────────────────────────────

def execute_global_phase_extraction(
    E           : float,
    numLinecuts : int,
    min_temp_range : float = MIN_TEMP_RANGE,
) -> None:
    T, nu_array, R = load_field(E)
    numCol = R.shape[1]

    selected_cols = (
        [numCol // 2] if numLinecuts == 1
        else [int(round(i * (numCol - 1) / (numLinecuts - 1))) for i in range(numLinecuts)]
    )

    print("\n─── PASS 1: SC pre-detection + baseline fits ───────────────")
    baseline_fits: Dict[int, dict] = {}

    for col in selected_cols:
        rho    = R[:, col]
        sc_mask, T_c = detect_superconducting(T, rho)

        # Segment only the *non-SC* portion of the curve
        T_fit   = T[~sc_mask]
        rho_fit = rho[~sc_mask]

        if len(T_fit) < 6:
            print(f"  ν={nu_array[col]:.3f} — almost entirely SC, skipping DP.")
            baseline_fits[col] = {
                "nu": nu_array[col], "rho": rho,
                "cost": 0.0, "segs": [],
                "sc_mask": sc_mask, "T_c": T_c,
                "neel_indices": [],
            }
            continue

        try:
            _, _, segs, cost = segment_linecut_dp_constrained(
                T_fit, rho_fit, min_temp_range=min_temp_range
            )
            segs, neel_idx = relabel_afm_segments(segs, T_fit)
            baseline_fits[col] = {
                "nu": nu_array[col], "rho": rho,
                "cost": cost, "segs": segs,
                "sc_mask": sc_mask, "T_c": T_c,
                "neel_indices": neel_idx,
            }
        except RuntimeError as e:
            print(f"  ν={nu_array[col]:.3f} — DP failed ({e}), skipping.")

    # Sort by confidence (lowest DP cost = highest Bayesian evidence)
    sorted_cols = sorted(baseline_fits.keys(), key=lambda c: baseline_fits[c]["cost"])

    print("\n─── PASS 2: Iterative propagation (confident-first) ────────")
    established_phases: Dict[int, List[SegmentFit]] = {}

    for col in sorted_cols:
        entry   = baseline_fits[col]
        nu      = entry["nu"]
        rho     = entry["rho"]
        sc_mask = entry["sc_mask"]
        T_c     = entry["T_c"]

        T_fit   = T[~sc_mask]
        rho_fit = rho[~sc_mask]

        # Nearest established neighbor as topological prior
        nearest_neighbor_segs = None
        if established_phases:
            nearest_col           = min(established_phases.keys(),
                                        key=lambda c: abs(nu_array[c] - nu))
            nearest_neighbor_segs = established_phases[nearest_col]

        print(f"  ν={nu:.3f} | T_c={T_c:.1f} K" if T_c else f"  ν={nu:.3f} | no SC",
              f"| neighbor prior: {nearest_neighbor_segs is not None}")

        if len(T_fit) < 6:
            established_phases[col] = entry["segs"]
            plot_behavior_fits(nu, T, rho, [], sc_mask, T_c, neel_indices=[])
            continue

        try:
            _, _, final_segs, _ = segment_linecut_dp_constrained(
                T_fit, rho_fit,
                min_temp_range=min_temp_range,
                neighbor_segments=nearest_neighbor_segs,
            )
            final_segs, neel_idx = relabel_afm_segments(final_segs, T_fit)
        except RuntimeError as e:
            print(f"    DP failed on pass 2 ({e}), using pass-1 result.")
            final_segs = entry["segs"]
            neel_idx   = entry["neel_indices"]

        established_phases[col] = final_segs
        plot_behavior_fits(nu, T, rho, final_segs, sc_mask, T_c,
                           neel_indices=neel_idx)

    print("\n✓ Done.")

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    execute_global_phase_extraction(E=103, numLinecuts=40, min_temp_range=3.0)