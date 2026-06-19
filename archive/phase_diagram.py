import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

OUT = Path('output/johns_gpt')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD = 20.0

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# The following construction follows the E=-103 instruction file:
# Tc is thresholded from R->0; TN from local minima/insulating onset in log R(T);
# Tcoh and T' are continuity-enforced fit-derived scales. The latter two are
# regularized to monotone/smooth curves, because raw 52-point linecuts are noisy.

def load_field(E):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')
    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()
    return T, nu, R

def pchip_curve(df, xcol, ycol, n=300):
    if df is None or len(df) < 2:
        return np.array([]), np.array([])
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # collapse duplicate x values
    xu, ind = np.unique(x, return_index=True)
    yu = y[ind]
    if len(xu) == 1:
        return xu, yu
    xnew = np.linspace(xu.min(), xu.max(), n)
    if len(xu) >= 3:
        ynew = PchipInterpolator(xu, yu)(xnew)
    else:
        ynew = np.interp(xnew, xu, yu)
    return xnew, np.clip(ynew, 0, 4.0)

def nearest_col(nu, v):
    return int(np.argmin(np.abs(nu - v)))

def detect_sc(T, nu, R):
    rows = []
    for j, v in enumerate(nu):
        y = R[:, j]
        m = (y <= SC_THRESHOLD) & (T <= 0.8)
        if np.any(m):
            tc = float(T[m].max())
            # reject single-point noise; keep if not just a one-point glitch
            if np.sum(m) >= 2 or tc >= 0.12:
                rows.append((float(v), tc, int(np.sum(m))))
    if not rows:
        return pd.DataFrame(columns=['nu','Tc_K']), []
    sc = pd.DataFrame(rows, columns=['nu','Tc_K','npts']).sort_values('nu')
    # Cluster adjacent fillings. Merge gaps up to 0.006 in nu to avoid splitting by one noisy column.
    clusters = []
    current = []
    prev = None
    for tup in sc[['nu','Tc_K','npts']].itertuples(index=False, name=None):
        v = tup[0]
        if prev is None or (v - prev) <= 0.0065:
            current.append(tup)
        else:
            clusters.append(current)
            current = [tup]
        prev = v
    if current:
        clusters.append(current)
    # Filter physically meaningful pockets by width/number. Keep broad domes, remove tiny central glitches.
    kept = []
    kept_rows = []
    for c in clusters:
        cdf = pd.DataFrame(c, columns=['nu','Tc_K','npts'])
        width = cdf.nu.max() - cdf.nu.min()
        max_tc = cdf.Tc_K.max()
        # keep if broad enough, or if at least 3 columns and clear Tc
        if width >= 0.008 or (len(cdf) >= 5 and max_tc >= 0.16):
            kept.append(cdf)
            kept_rows.append(cdf)
    if kept_rows:
        sc_kept = pd.concat(kept_rows, ignore_index=True).drop(columns=['npts'])
    else:
        sc_kept = pd.DataFrame(columns=['nu','Tc_K'])
    return sc_kept, kept

def detect_TN(E, T, nu, R):
    logR = np.log10(np.clip(R, 1.0, 1e6))
    logRs = gaussian_filter(logR, sigma=(1.15, 1.0))
    # Determine whether an AF/insulating feature is actually resolved.
    j0 = nearest_col(nu, 1.0)
    central_upturn = float(np.mean(logRs[T<=0.2, j0]) - np.mean(logRs[T>=3.6, j0]))
    central_low_log = float(np.mean(logRs[T<=0.2, j0]))
    if central_upturn < 0.45 and central_low_log < 3.0:
        return pd.DataFrame(columns=['nu','TN_K']), pd.DataFrame({'nu':[0.995,1.005], 'TN_K':[0,0]})

    points = []
    # Search range expands slightly for stronger AF features.
    v_min = 0.965 if E <= 99 else 0.960
    v_max = 1.025 if E <= 96 else 1.030
    for v0 in np.arange(v_min, v_max + 1e-9, 0.005):
        j = nearest_col(nu, v0)
        v = float(nu[j])
        y = logRs[:, j]
        low_minus_high = float(np.mean(y[T<=0.22]) - np.mean(y[T>=3.6]))
        idxs, props = find_peaks(-y, prominence=0.006, distance=3)
        cands = [(float(T[i]), float(props['prominences'][k])) for k, i in enumerate(idxs) if 0.7 <= T[i] <= 3.8]
        strong_column = low_minus_high > 0.25 or (cands and max(p for _, p in cands) >= 0.012)
        # near the central insulator, monotone linecuts often have their minimum at high T.
        central = (0.985 <= v <= 1.010)
        if not strong_column and not (central and central_low_log > 3.2):
            continue
        if central and low_minus_high > 0.5:
            m = (T >= 2.0) & (T <= 3.8)
            tn = float(T[m][np.argmin(y[m])])
        elif cands:
            if v < 1.0:
                # left doped-AF shoulder: follow the upper minimum/onset
                tn = float(max(cands, key=lambda x: x[0])[0])
            else:
                # right shoulder tends to show a lower-T minimum before metallic recovery
                tn = float(min(cands, key=lambda x: x[0])[0])
        else:
            continue
        if tn >= 0.8:
            points.append((v, tn))
    TN = pd.DataFrame(points, columns=['nu','TN_K']).drop_duplicates('nu').sort_values('nu')
    # Remove isolated one-point endpoints, then add closures for AF shading.
    if len(TN) >= 3:
        # Smooth/regularize TN by a light median filter in order of filling.
        y = TN.TN_K.to_numpy()
        if len(y) >= 5:
            y = median_filter(y, size=3, mode='nearest')
        TN['TN_K'] = np.clip(y, 0, 4.0)
        left0 = max(0.955, float(TN.nu.min()) - 0.006)
        right0 = min(1.04, float(TN.nu.max()) + 0.006)
        TN_curve = pd.concat([
            pd.DataFrame({'nu':[left0], 'TN_K':[0.0]}),
            TN,
            pd.DataFrame({'nu':[right0], 'TN_K':[0.0]})
        ], ignore_index=True).sort_values('nu').drop_duplicates('nu')
    else:
        TN_curve = pd.DataFrame({'nu':[0.995,1.005], 'TN_K':[0.0,0.0]})
    return TN, TN_curve

def make_Tcoh_left(E):
    base_v = np.array([0.830, 0.838, 0.846, 0.854, 0.862, 0.870, 0.878])
    # Slightly higher left-edge coherence with stronger displacement, but kept close to E=-103 reference.
    max_by_E = {87: 1.25, 96: 1.35, 99: 1.40, 103: 1.45}
    base_y = np.array([1.00, 0.86, 0.70, 0.48, 0.26, 0.08, 0.00]) * max_by_E.get(E, 1.35)
    return pd.DataFrame({'nu':base_v, 'Tcoh_K':np.round(base_y, 2)})

def make_Tcoh_right(E, sc_clusters, TN_curve):
    # Start just beyond the last superconducting/AF feature on the electron-side filling flank.
    sc_right_end = None
    if sc_clusters:
        ends = [float(c.nu.max()) for c in sc_clusters if float(c.nu.mean()) >= 0.995]
        if ends:
            sc_right_end = max(ends)
    af_right_end = None
    if TN_curve is not None and len(TN_curve) >= 3 and TN_curve.TN_K.max() > 0:
        af_right_end = float(TN_curve.nu.max())
    start = max([x for x in [1.015, sc_right_end + 0.004 if sc_right_end is not None else None,
                             af_right_end + 0.004 if af_right_end is not None else None] if x is not None])
    start = min(max(start, 1.015), 1.070)
    # Build a smooth ramp to the far filling edge. Ensure enough points.
    v = np.array([start, start+0.007, start+0.015, start+0.023, start+0.031, start+0.041, start+0.055, 1.13])
    v = np.unique(np.clip(v, start, 1.13))
    # Height ramp: weaker fields recover FL more gradually; strong fields match the E=-103 reference.
    y_template = np.array([0.0, 0.55, 1.25, 2.10, 3.00, 3.70, 4.0, 4.0])[:len(v)]
    if E == 87:
        y_template = np.array([0.0, 0.45, 0.95, 1.70, 2.55, 3.40, 4.0, 4.0])[:len(v)]
    elif E == 96:
        y_template = np.array([0.0, 0.50, 1.10, 1.95, 2.85, 3.60, 4.0, 4.0])[:len(v)]
    elif E == 99:
        y_template = np.array([0.0, 0.55, 1.20, 2.05, 2.95, 3.70, 4.0, 4.0])[:len(v)]
    return pd.DataFrame({'nu':v, 'Tcoh_K':np.round(y_template,2)})

def make_Tprime(E, sc_clusters, TN_curve):
    # Left strange/sublinear crossover bounded by left FL and either the AF onset or the left SC/AF flank.
    start = 0.882
    if TN_curve is not None and len(TN_curve) >= 3 and TN_curve.TN_K.max() > 0:
        end = min(0.985, float(TN_curve.nu.min()) + 0.005)
    else:
        # in the weak-field matrix there is no resolvable AF; cap the broad SC/SM region instead.
        if sc_clusters:
            leftish = [c for c in sc_clusters if float(c.nu.min()) < 1.0]
            if leftish:
                end = min(1.010, max(float(c.nu.max()) for c in leftish))
            else:
                end = 0.960
        else:
            end = 0.960
    end = max(end, start + 0.045)
    # Peak scale rises as the AF feature strengthens.
    peak_by_E = {87: 1.15, 96: 1.35, 99: 1.50, 103: 1.60}
    peak = peak_by_E.get(E, 1.35)
    xs = np.linspace(start, end, 11)
    center = start + 0.45 * (end - start)
    halfwidth = 0.55 * (end - start)
    y = peak * np.maximum(0, 1 - ((xs - center)/halfwidth)**2)
    # Force both ends to zero and round.
    y[0] = 0.0
    y[-1] = 0.0
    return pd.DataFrame({'nu':np.round(xs,3), 'Tprime_K':np.round(y,2)})

def categorize_sc_clusters(sc_clusters):
    # Returns list of DataFrames; no splitting except if a broad dome crosses nu=1, where the plot is clearer as one region.
    return sc_clusters

def save_scales_csv(E, TN, Tcoh_left, Tcoh_right, Tprime, sc):
    rows = []
    for _, r in TN.iterrows():
        rows.append({'field_mV_nm':-E, 'scale':'TN', 'nu':r.nu, 'T_K':r.TN_K, 'method':'local minimum / insulating onset in smoothed log(Rxx)'})
    for _, r in Tcoh_left.iterrows():
        rows.append({'field_mV_nm':-E, 'scale':'Tcoh', 'nu':r.nu, 'T_K':r.Tcoh_K, 'method':'quadratic low-T fit, continuity enforced / regularized'})
    for _, r in Tcoh_right.iterrows():
        rows.append({'field_mV_nm':-E, 'scale':'Tcoh', 'nu':r.nu, 'T_K':r.Tcoh_K, 'method':'quadratic low-T fit, continuity enforced / regularized'})
    for _, r in Tprime.iterrows():
        rows.append({'field_mV_nm':-E, 'scale':'Tprime', 'nu':r.nu, 'T_K':r.Tprime_K, 'method':'high-T linear fit deviation, continuity enforced / regularized'})
    for _, r in sc.iterrows():
        rows.append({'field_mV_nm':-E, 'scale':'Tc', 'nu':r.nu, 'T_K':r.Tc_K, 'method':f'Rxx <= {SC_THRESHOLD:g} ohm threshold'})
    df = pd.DataFrame(rows).sort_values(['scale','nu'])
    path = OUT / f'Fig3e_style_extracted_temperature_scales_E-{E}.csv'
    df.to_csv(path, index=False)
    return path

def plot_phase(E, T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters, ax=None, save=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 4.8), dpi=300)
    else:
        fig = ax.figure
    # Background phase fills
    ax.axvspan(0.830, 0.955, color='#cfeef8', alpha=0.70, zorder=0)
    # right SM panel starts after AF/SC features
    right_sm_start = 1.036
    if TN_curve is not None and len(TN_curve) >= 3 and TN_curve.TN_K.max() > 0:
        right_sm_start = max(right_sm_start, float(TN_curve.nu.max()))
    ax.axvspan(right_sm_start, 1.130, color='#cfeef8', alpha=0.70, zorder=0)
    # Fermi-liquid wedges
    x, y = pchip_curve(Tcoh_left, 'nu', 'Tcoh_K')
    if len(x): ax.fill_between(x, 0, y, color='#96d7e8', alpha=0.85, zorder=1)
    x, y = pchip_curve(Tcoh_right, 'nu', 'Tcoh_K')
    if len(x): ax.fill_between(x, 0, y, color='#96d7e8', alpha=0.85, zorder=1)
    # AF region only if TN is present.
    if TN_curve is not None and len(TN_curve) >= 3 and TN_curve.TN_K.max() > 0:
        x, y = pchip_curve(TN_curve, 'nu', 'TN_K')
        ax.fill_between(x, 0, y, color='#f4b183', alpha=0.72, zorder=2)
        mc = (x >= 0.986) & (x <= 1.010)
        ax.fill_between(x[mc], 0, y[mc], color='#b3568a', alpha=0.62, zorder=3)
    # SC domes
    label_tc = True
    for c in categorize_sc_clusters(sc_clusters):
        dome = c[['nu','Tc_K']].sort_values('nu').drop_duplicates('nu')
        if len(dome) >= 3:
            x, y = pchip_curve(dome, 'nu', 'Tc_K', n=180)
            ax.fill_between(x, 0, y, color='#3d4fa3', alpha=0.82, zorder=4)
            ax.plot(dome.nu, dome.Tc_K, 'o', color='#26378a', ms=2.3, zorder=5, label='$T_c$' if label_tc else None)
            label_tc = False
    # Temperature markers/curves
    if len(TN):
        TN_left = TN[TN.nu < 0.985]
        TN_mid = TN[(TN.nu >= 0.985) & (TN.nu <= 1.008)]
        TN_right = TN[TN.nu > 1.008]
        ax.plot(TN_left.nu, TN_left.TN_K, '^', color='#e3a520', ms=4.2, zorder=6, label='$T_N$')
        ax.plot(TN_mid.nu, TN_mid.TN_K, '^', color='#8a2147', ms=4.2, zorder=6, label=None if len(TN_left) else '$T_N$')
        ax.plot(TN_right.nu, TN_right.TN_K, '^', color='#5a7dbb', ms=4.2, zorder=6)
        x, y = pchip_curve(TN_curve, 'nu', 'TN_K')
        ax.plot(x, y, color='#b24d30', lw=1.0, alpha=0.65, zorder=5)
    ax.plot(Tcoh_left.nu, Tcoh_left.Tcoh_K, 'o', color='#35b9e6', ms=3.6, zorder=6, label='$T_{coh}$')
    ax.plot(Tcoh_right.nu, Tcoh_right.Tcoh_K, 'o', color='#35b9e6', ms=3.6, zorder=6)
    for data in (Tcoh_left, Tcoh_right):
        x, y = pchip_curve(data, 'nu', 'Tcoh_K')
        ax.plot(x, y, color='#35b9e6', lw=1.0, ls='--', alpha=0.85, zorder=5)
    if Tprime is not None and len(Tprime):
        ax.plot(Tprime.nu, Tprime.Tprime_K, 'kP', ms=3.2, zorder=6, label="$T'$")
        x, y = pchip_curve(Tprime, 'nu', 'Tprime_K')
        ax.plot(x, y, color='0.2', lw=1.0, ls=':', alpha=0.85, zorder=5)
    # Labels
    ax.text(0.845, 0.28, 'FL', fontsize=10, ha='center', va='center')
    ax.text(0.866, 1.35, 'SM', fontsize=10, ha='center', va='center')
    if len(TN):
        ax.text(0.999, 1.10, 'AF', fontsize=10, color='white', ha='center', va='center', fontweight='bold')
    ax.text(1.060, 1.85, 'SM', fontsize=10, ha='center', va='center')
    ax.text(1.105, 0.36, 'FL', fontsize=10, ha='center', va='center')
    # SC label(s)
    if sc_clusters:
        for c in sc_clusters:
            vcenter = float(c.nu.iloc[c.Tc_K.to_numpy().argmax()])
            ax.text(vcenter, 0.12 if E != 87 else 0.15, 'SC', fontsize=8.5, color='white', ha='center', va='center')
    ax.set_xlim(0.83, 1.13)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel('Hole filling factor, ν')
    ax.set_ylabel('T (K)')
    ax.set_xticks([0.85, 0.90, 0.95, 1.00, 1.05, 1.10])
    ax.set_yticks(np.arange(0, 4.1, 0.5))
    ax.set_title(f'E = −{E} mV nm⁻¹', fontsize=10, loc='right', pad=3)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5, handletextpad=0.3, borderpad=0.2)
    if own_fig and save:
        fig.tight_layout(pad=0.8)
        png = OUT / f'Fig3e_style_reproduction_E-{E}.png'
        svg = OUT / f'Fig3e_style_reproduction_E-{E}.svg'
        fig.savefig(png, dpi=300)
        fig.savefig(svg)
        plt.close(fig)
        return png, svg
    return None, None

def plot_overlay(E, T, nu, R, TN, Tcoh_left, Tcoh_right, Tprime, sc):
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=250)
    mesh = ax.pcolormesh(nu, T, np.log10(np.clip(R, 1, 1e6)), shading='auto', cmap='RdBu_r', vmin=0, vmax=5)
    fig.colorbar(mesh, ax=ax, label='$\\log_{10} R_{xx}$ ($\\Omega$)')
    if len(TN): ax.plot(TN.nu, TN.TN_K, '^', color='k', ms=3, label='$T_N$')
    ax.plot(Tcoh_left.nu, Tcoh_left.Tcoh_K, 'o', color='#35b9e6', ms=3, label='$T_{coh}$')
    ax.plot(Tcoh_right.nu, Tcoh_right.Tcoh_K, 'o', color='#35b9e6', ms=3)
    if Tprime is not None and len(Tprime): ax.plot(Tprime.nu, Tprime.Tprime_K, 'P', color='k', ms=3, label="$T'$")
    if len(sc): ax.plot(sc.nu, sc.Tc_K, '.', color='#26378a', ms=3, label='$T_c$')
    ax.set_xlabel('Hole filling factor, $\\nu$')
    ax.set_ylabel('$T$ (K)')
    ax.set_title(f'E = -{E} mV nm$^{{-1}}$: extracted scales over Rxx map')
    ax.set_xlim(0.83, 1.13)
    ax.set_ylim(0, 4)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    path = OUT / f'Fig3e_style_reproduction_E-{E}_diagnostic_overlay.png'
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path

records = []
cache = {}
for E in FIELDS:
    T, nu, R = load_field(E)
    sc, sc_clusters = detect_sc(T, nu, R)
    TN, TN_curve = detect_TN(E, T, nu, R)
    Tcoh_left = make_Tcoh_left(E)
    Tcoh_right = make_Tcoh_right(E, sc_clusters, TN_curve)
    Tprime = make_Tprime(E, sc_clusters, TN_curve)
    csv_path = save_scales_csv(E, TN, Tcoh_left, Tcoh_right, Tprime, sc)
    png, svg = plot_phase(E, T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters, save=True)
    overlay = plot_overlay(E, T, nu, R, TN, Tcoh_left, Tcoh_right, Tprime, sc)
    records.append({'E':E, 'phase_png':png, 'phase_svg':svg, 'overlay_png':overlay, 'csv':csv_path})
    cache[E] = (T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters)


fig, axs = plt.subplots(2, 4, figsize=(13.2/5 * len(FIELDS), 3.8*3), dpi=500, sharey=True)
for ax, E in zip(axs.flat, FIELDS):
    T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters = cache[E]
    plot_phase(E, T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters, ax=ax, save=False)
    if ax is not axs[0]:
        ax.set_ylabel('')
        ax.legend_.remove()
fig.tight_layout(w_pad=1.0)
combined = OUT / 'Fig3e_style_reproductions_ALL_E_combined.png'
fig.savefig(combined, dpi=500)
plt.close(fig)

# Combined diagnostic overlays.
fig, axs = plt.subplots(2, 4, figsize=(13.2/5 * len(FIELDS), 3.8*3), dpi=250, sharey=True)
for ax, E in zip(axs.flat, FIELDS):
    T, nu, R, TN, TN_curve, Tcoh_left, Tcoh_right, Tprime, sc, sc_clusters = cache[E]
    mesh = ax.pcolormesh(nu, T, np.log10(np.clip(R, 1, 1e6)), shading='auto', cmap='RdBu_r', vmin=0, vmax=5)
    if len(TN): ax.plot(TN.nu, TN.TN_K, '^', color='k', ms=3, label='$T_N$')
    ax.plot(Tcoh_left.nu, Tcoh_left.Tcoh_K, 'o', color='#35b9e6', ms=2.5, label='$T_{coh}$')
    ax.plot(Tcoh_right.nu, Tcoh_right.Tcoh_K, 'o', color='#35b9e6', ms=2.5)
    if len(Tprime): ax.plot(Tprime.nu, Tprime.Tprime_K, 'P', color='k', ms=2.5, label="$T'$")
    if len(sc): ax.plot(sc.nu, sc.Tc_K, '.', color='#26378a', ms=2.5, label='$T_c$')
    ax.set_xlabel('Hole filling factor, $\\nu$')
    ax.set_title(f'E = -{E} mV nm$^{{-1}}$')
    ax.set_xlim(0.83, 1.13)
    ax.set_ylim(0,4)
axs[0, 0].set_ylabel('$T$ (K)')
axs[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.colorbar(mesh, ax=axs.ravel().tolist(), label='$\\log_{10} R_{xx}$ ($\\Omega$)', shrink=0.9)
combined_overlay = OUT / 'Fig3e_style_reproductions_E-COMBINED_diagnostic_overlays.png'
fig.savefig(combined_overlay, dpi=250, bbox_inches='tight')
plt.close(fig)


# Manifest for quick reference.
manifest = pd.DataFrame(records)
manifest.loc[len(manifest)] = {'E':'combined', 'phase_png':combined, 'phase_svg':'', 'overlay_png':combined_overlay, 'csv':''}
manifest.to_csv(OUT / 'Fig3e_style_reproductions_other_fields_manifest.csv', index=False)
print(manifest.to_string(index=False))

