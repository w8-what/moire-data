import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm

from moire.extract_power_law import _fit as fit_power_law, extract_local_fits


def _hill_sigmoid(x, reference_value, reference_score=0.8, coeff=2):

    C = reference_value**coeff * (1 - reference_score) / reference_score
    return x**coeff / (x**coeff + C)


def extract_upturns(T, linecut, min_pts=5, min_width=0.5, sigma=5, coeff=2) -> list[dict]:

    candidate_upturns = []
    rho_smoothed = linecut.get("rho_smoothed")
    noise = linecut.get("local_noise")

    peaks, prop = find_peaks(-rho_smoothed, prominence=(None, None), height=(None, None))

    for i, idx in enumerate(peaks):

        # Finding horizontal persistence
        left_base_idx = prop["left_bases"][i]
        right_base_idx = prop["right_bases"][i]

        rho_horizontal = min(rho_smoothed[right_base_idx], rho_smoothed[left_base_idx])

        if rho_smoothed[right_base_idx] - rho_smoothed[left_base_idx] > 0:
            # use right as point and find right point that corresponds to right
            j = idx + 1  # watch out edge
            while j <= right_base_idx:
                low = min(rho_smoothed[j - 1], rho_smoothed[j])
                high = max(rho_smoothed[j - 1], rho_smoothed[j])

                if low <= rho_horizontal and rho_horizontal <= high:
                    break

                j += 1

            left_idx = left_base_idx
            right_idx = j

        else:
            # use left as point and find right point that corresponds to right
            j = idx - 1  # watch out edge
            while j >= left_base_idx:
                low = min(rho_smoothed[j + 1], rho_smoothed[j])
                high = max(rho_smoothed[j + 1], rho_smoothed[j])

                if low <= rho_horizontal and rho_horizontal <= high:
                    break

                j -= 1

            left_idx = j
            right_idx = right_base_idx

        local_noise = np.mean(noise[left_idx : right_idx + 1])
        prominence = prop.get("prominences")[i]
        prom_z = prominence / local_noise

        width = T[right_idx] - T[left_idx]
        pts = len(T[left_idx : right_idx + 1])

        target = 0.8

        C_prom = sigma**coeff * (1 - target) / target
        C_width = min_width**coeff * (1 - target) / target
        C_pts = min_pts**coeff * (1 - target) / target

        prom_score = _hill_sigmoid(prom_z, sigma, target, 2)
        pts_score = pts**coeff / (pts**coeff + C_pts)
        pts_score = _hill_sigmoid(pts, min_pts)
        width_score = width**coeff / (width**coeff + C_width)

        comb_score = prom_score**0.5 * pts_score**0.3 * width_score**0.2
        comb_score = float(f"{comb_score:.3g}")

        feature = {"T": T[idx], "nu": linecut.get("nu"), "type": "upturn", "confidence": comb_score}

        candidate_upturns.append(feature)

    return candidate_upturns


def extract_downturns(T, linecut, min_pts=5, min_width=0.5, sigma=5, coeff=2) -> list[dict]:

    candidate_downturns = []
    rho_smoothed = linecut.get("rho_smoothed")
    noise = linecut.get("local_noise")

    peaks, prop = find_peaks(rho_smoothed, prominence=(None, None), height=(None, None))

    for i, idx in enumerate(peaks):

        # Finding horizontal persistence
        left_base_idx = prop["left_bases"][i]
        right_base_idx = prop["right_bases"][i]

        rho_horizontal = max(rho_smoothed[right_base_idx], rho_smoothed[left_base_idx])

        if rho_smoothed[right_base_idx] - rho_smoothed[left_base_idx] > 0:
            # use right as point and find left point that corresponds to right
            j = idx - 1  # watch out edge
            while j >= left_base_idx:
                low = min(rho_smoothed[j + 1], rho_smoothed[j])
                high = max(rho_smoothed[j + 1], rho_smoothed[j])

                if low <= rho_horizontal and rho_horizontal <= high:
                    break

                j -= 1

            left_idx = j
            right_idx = right_base_idx

        else:
            # use left as point and find right point that corresponds to right
            j = idx + 1  # watch out edge
            while j <= right_base_idx:
                low = min(rho_smoothed[j - 1], rho_smoothed[j])
                high = max(rho_smoothed[j - 1], rho_smoothed[j])

                if low <= rho_horizontal and rho_horizontal <= high:
                    break

                j += 1

            left_idx = left_base_idx
            right_idx = j

        local_noise = np.mean(noise[left_idx : right_idx + 1])
        prominence = prop.get("prominences")[i]
        prom_z = prominence / local_noise

        width = T[right_idx] - T[left_idx]
        pts = len(T[left_idx : right_idx + 1])

        target = 0.8

        C_prom = sigma**coeff * (1 - target) / target
        C_width = min_width**coeff * (1 - target) / target
        C_pts = min_pts**coeff * (1 - target) / target

        prom_score = prom_z**coeff / (prom_z**coeff + C_prom)
        pts_score = pts**coeff / (pts**coeff + C_pts)
        width_score = width**coeff / (width**coeff + C_width)

        comb_score = prom_score**0.5 * pts_score**0.3 * width_score**0.2
        comb_score = float(f"{comb_score:.3g}")

        feature = {
            "T": T[idx],
            "nu": linecut.get("nu"),
            "type": "downturn",
            "confidence": comb_score,
        }

        candidate_downturns.append(feature)

    return candidate_downturns


def extract_Tc(T, linecut, threshold=20, max_candidates=3) -> list[dict]:

    # find each point that are below the resistivity threshold
    # for each point calculate the following
    # 1. the temp fraction that is below the resistivity threshold
    # 2. the number of points under the temperature
    # 3. the temperature range of the threshold
    # 4. use geometric mean for scoring

    candidate_Tcs = []

    rho = linecut.get("rho")
    below = rho <= threshold

    for idx in np.flatnonzero(below):

        T_lower = T[: idx + 1]
        below_lower = rho[: idx + 1] < threshold

        num_points = np.count_nonzero(below_lower)  # not necessairely all prior points are below
        temp_range = np.trapezoid(
            below_lower.astype(float), T_lower
        )  # not nessairely all prior points are below either

        total_range = T_lower[-1] - T_lower[0]
        temp_frac = temp_range / total_range if total_range > 0 else 0.0

        score_pts = _hill_sigmoid(num_points, 5)
        score_temp = _hill_sigmoid(temp_range, 0.5)
        score_frac = _hill_sigmoid(temp_frac, 0.9)

        comb_score = score_frac ** (1 / 3) * score_pts ** (1 / 3) * score_temp ** (1 / 3)

        feature = {"T": T[idx], "nu": linecut.get("nu"), "type": "Tc", "confidence": comb_score}

        candidate_Tcs.append(feature)

    candidate_Tcs.sort(key=lambda feature: feature["confidence"], reverse=True)

    return candidate_Tcs[:max_candidates]


def extract_Tcoh(
    T,
    linecut,
    max_candidates=3,
    deviation=0.10,
    min_n_probability=0.8,
    min_fit_points=6,
    min_fit_span=0.5,
    persistence=0.5,
) -> list[dict]:
    """Find stable 10%-departure points from a low-T quadratic fit."""

    # Use the raw data for fitting, the smoothed data for locating the
    # crossover, and the local noise to decide whether a deviation is real.
    T = np.asarray(T, float)
    rho = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", rho), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float)
    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    sigma[~valid_sigma] = np.median(sigma[valid_sigma]) if np.any(valid_sigma) else 1.0

    # Respect a manually/automatically supplied extraction range when one is
    # available; otherwise search the whole measured temperature range.
    lower, upper = T[0], T[-1]
    for behavior in linecut.get("behaviors", []):
        if behavior.get("type") == "extraction_range":
            lower, upper = sorted((behavior["T_lower"], behavior["T_upper"]))
            break

    allowed = np.flatnonzero((T >= lower) & (T <= upper))
    if len(allowed) < min_fit_points + 2:
        return []

    # Try several possible endpoints for the low-T fit. A genuine crossover
    # should be found by more than one reasonable choice of fitting window.
    first, last = allowed[0], allowed[-1]
    fit_end_temperatures = np.arange(T[first] + min_fit_span, T[last] - persistence, 0.01)
    fit_ends = np.unique([np.argmin(abs(T - value)) for value in fit_end_temperatures])
    fit_ends = [
        end
        for end in fit_ends
        if end - first + 1 >= min_fit_points and T[end] - T[first] >= min_fit_span
    ]

    votes = {}
    valid_windows = 0
    for end in fit_ends:
        selection = slice(first, end + 1)

        # Rough screen: fit rho = rho0 + A*T**n and keep this window only when
        # its fitted n and uncertainty give enough probability to n ~= 2.
        rough_fit = fit_power_law(T[selection], rho[selection], sigma[selection], (0.2, 4.0))
        if rough_fit is None or not np.isfinite(rough_fit["n_sigma"]):
            continue

        if rough_fit["n_sigma"] == 0:
            n_probability = float(1.5 <= rough_fit["n"] <= 2.5)
        else:
            n_probability = norm.cdf((2.5 - rough_fit["n"]) / rough_fit["n_sigma"]) - norm.cdf(
                (1.5 - rough_fit["n"]) / rough_fit["n_sigma"]
            )
        if n_probability < min_n_probability:
            continue

        # Once the rough screen says the window is plausibly quadratic, refit
        # the simpler physical baseline rho = rho0 + A*T**2.
        design = np.column_stack((np.ones(end - first + 1), T[selection] ** 2))
        weighted_design = design / sigma[selection, None]
        rho0, coefficient = np.linalg.lstsq(
            weighted_design, rho[selection] / sigma[selection], rcond=None
        )[0]
        if coefficient <= 0:
            continue
        valid_windows += 1

        # A point is outside the coherent T**2 regime only if the departure is
        # both large in relative terms and larger than the estimated noise.
        predicted = rho0 + coefficient * T**2
        difference = abs(smooth - predicted)
        exceeds = (difference / np.maximum(abs(predicted), 1e-12) >= deviation) & (
            difference / sigma >= 1.5
        )

        # Take the first departure that persists over a finite temperature
        # interval. This avoids calling a single noisy point T_coh.
        for idx in range(end + 1, last + 1):
            stop = np.searchsorted(T, T[idx] + persistence)
            if stop <= last and np.mean(exceeds[idx : stop + 1]) >= 0.8:
                # Windows vote at exact measured temperatures; nearby
                # temperatures remain separate candidates for the later DP.
                votes[idx] = votes.get(idx, 0.0) + n_probability
                break

    if not votes or not valid_windows:
        return []

    # Turn agreement across valid fit windows into the current support score.
    # This is not yet a calibrated probability that T is the true T_coh.
    window_score = min(1.0, valid_windows / 3)
    candidates = [
        {
            "T": float(T[idx]),
            "nu": linecut.get("nu"),
            "type": "Tcoh",
            "confidence": float(f"{window_score * vote / valid_windows:.3g}"),
        }
        for idx, vote in votes.items()
    ]

    candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
    return candidates


def extract_Tcoh_best_fits(
    T,
    linecut,
    max_candidates=5,
    deviation=0.10,
    min_n_probability=0.8,
    min_fit_points=6,
    min_fit_span=0.5,
    persistence=0.5,
    persistence_fraction=0.8,
) -> list[dict]:
    """Return Tcoh candidates from the best independently ranked T² fits."""
    T = np.asarray(T, float)
    rho = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", rho), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float).copy()

    # Replace unusable noise estimates before doing either weighted fit.
    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    sigma[~valid_sigma] = np.median(sigma[valid_sigma]) if np.any(valid_sigma) else 1.0

    # Anchor every candidate window to the bottom of the extraction range.
    lower, upper = T[0], T[-1]
    for behavior in linecut.get("behaviors", []):
        if behavior.get("type") == "extraction_range":
            lower, upper = sorted((behavior["T_lower"], behavior["T_upper"]))
            break

    allowed = np.flatnonzero((T >= lower) & (T <= upper))
    if len(allowed) < min_fit_points + 1:
        return []

    first, last = allowed[0], allowed[-1]
    quadratic_fits = []

    # Screen each expanding window with a free exponent, then refit accepted
    # windows using the physical rho = rho0 + A*T² model.
    for end in allowed[min_fit_points - 1 :]:
        if T[end] - T[first] < min_fit_span:
            continue
        if np.searchsorted(T, T[end] + persistence, side="left") > last:
            break

        selection = slice(first, end + 1)
        rough_fit = fit_power_law(
            T[selection], rho[selection], sigma[selection], (0.2, 4.0)
        )
        if rough_fit is None or not np.isfinite(rough_fit["n_sigma"]):
            continue

        if rough_fit["n_sigma"] == 0:
            n_probability = float(1.5 <= rough_fit["n"] <= 2.5)
        else:
            n_probability = norm.cdf(
                (2.5 - rough_fit["n"]) / rough_fit["n_sigma"]
            ) - norm.cdf((1.5 - rough_fit["n"]) / rough_fit["n_sigma"])
        if n_probability < min_n_probability:
            continue

        design = np.column_stack((np.ones(end - first + 1), T[selection] ** 2))
        weighted_design = design / sigma[selection, None]
        rho0, coefficient = np.linalg.lstsq(
            weighted_design, rho[selection] / sigma[selection], rcond=None
        )[0]
        if coefficient <= 0:
            continue

        fitted = rho0 + coefficient * T[selection] ** 2
        residuals = (rho[selection] - fitted) / sigma[selection]
        degrees_of_freedom = len(residuals) - 2
        reduced_chi2 = np.sum(residuals**2) / degrees_of_freedom
        if not np.isfinite(reduced_chi2):
            continue

        quadratic_fits.append(
            {
                "end": end,
                "rho0": float(rho0),
                "A": float(coefficient),
                "n": rough_fit["n"],
                "n_sigma": rough_fit["n_sigma"],
                "n_probability": float(n_probability),
                "reduced_chi2": float(reduced_chi2),
            }
        )

    # Lower reduced chi-squared is better; prefer the longer fit on exact ties.
    quadratic_fits.sort(
        key=lambda fit: (fit["reduced_chi2"], -(T[fit["end"]] - T[first]))
    )

    candidates = []
    for fit in quadratic_fits:
        predicted = fit["rho0"] + fit["A"] * T**2
        difference = abs(smooth - predicted)
        exceeds = (difference / np.maximum(abs(predicted), 1e-12) >= deviation) & (
            difference / sigma >= 1.5
        )

        # Find the first 10% departure that persists over the requested
        # temperature span. The fraction is integrated over temperature so the
        # result is not biased by uneven sampling density.
        transition_idx = None
        for idx in range(fit["end"] + 1, last + 1):
            stop = np.searchsorted(T, T[idx] + persistence, side="left")
            if stop > last:
                break

            local_T = T[idx : stop + 1]
            local_exceeds = exceeds[idx : stop + 1].astype(float)
            persistent_fraction = np.trapezoid(local_exceeds, local_T) / np.ptp(
                local_T
            )
            if persistent_fraction >= persistence_fraction:
                transition_idx = idx
                break

        if transition_idx is None:
            continue

        candidates.append(
            {
                "T": float(T[transition_idx]),
                "nu": linecut.get("nu"),
                "type": "Tcoh",
                "confidence": float(1.0 / (1.0 + fit["reduced_chi2"])),
                "fit_T_lower": float(T[first]),
                "fit_T_upper": float(T[fit["end"]]),
                "n": float(fit["n"]),
                "n_sigma": float(fit["n_sigma"]),
                "n_probability": fit["n_probability"],
                "reduced_chi2": fit["reduced_chi2"],
            }
        )
        if len(candidates) == max_candidates:
            break

    return candidates


def extract_Tcoh_direct_fits(
    T,
    linecut,
    max_candidates=5,
    deviation=0.10,
    min_fit_points=6,
    min_fit_span=0.5,
    max_reduced_chi2=3.0,
) -> list[dict]:
    """Return direct 10%-departure candidates from the best T² fits."""
    T = np.asarray(T, float)
    rho = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", rho), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float).copy()

    # Replace invalid noise estimates before the weighted T² fits.
    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    sigma[~valid_sigma] = np.median(sigma[valid_sigma]) if np.any(valid_sigma) else 1.0

    # Anchor every fit to the bottom of the existing extraction range.
    lower, upper = T[0], T[-1]
    for behavior in linecut.get("behaviors", []):
        if behavior.get("type") == "extraction_range":
            lower, upper = sorted((behavior["T_lower"], behavior["T_upper"]))
            break

    allowed = np.flatnonzero((T >= lower) & (T <= upper))
    if len(allowed) < min_fit_points + 1:
        return []

    first, last = allowed[0], allowed[-1]
    quadratic_fits = []

    # Fit every expanding low-temperature window directly to rho0 + A*T².
    # This avoids rejecting short windows whose free exponent is uncertain.
    for end in allowed[min_fit_points - 1 : -1]:
        if T[end] - T[first] < min_fit_span:
            continue

        selection = slice(first, end + 1)
        design = np.column_stack((np.ones(end - first + 1), T[selection] ** 2))
        weighted_design = design / sigma[selection, None]
        rho0, coefficient = np.linalg.lstsq(
            weighted_design, rho[selection] / sigma[selection], rcond=None
        )[0]
        if coefficient <= 0:
            continue

        fitted = rho0 + coefficient * T[selection] ** 2
        residuals = (rho[selection] - fitted) / sigma[selection]
        reduced_chi2 = np.sum(residuals**2) / (len(residuals) - 2)
        if not np.isfinite(reduced_chi2) or reduced_chi2 > max_reduced_chi2:
            continue

        quadratic_fits.append(
            {
                "end": end,
                "rho0": float(rho0),
                "A": float(coefficient),
                "reduced_chi2": float(reduced_chi2),
            }
        )

    # Select the requested best fits before looking for a departure. This
    # prevents lower-quality short fits from creating false candidates.
    quadratic_fits.sort(
        key=lambda fit: (
            fit["reduced_chi2"],
            -(T[fit["end"]] - T[first]),
        )
    )

    candidates = []
    for fit in quadratic_fits[:max_candidates]:
        # Apply the paper's 10% criterion directly to the smoothed linecut.
        predicted = fit["rho0"] + fit["A"] * T**2
        relative_difference = abs(smooth - predicted) / np.maximum(
            abs(predicted), 1e-12
        )
        departures = np.flatnonzero(
            relative_difference[fit["end"] + 1 : last + 1] >= deviation
        )
        if not len(departures):
            continue
        transition_idx = fit["end"] + 1 + departures[0]

        candidates.append(
            {
                "T": float(T[transition_idx]),
                "nu": linecut.get("nu"),
                "type": "Tcoh",
                "confidence": float(1.0 / (1.0 + fit["reduced_chi2"])),
                "fit_T_lower": float(T[first]),
                "fit_T_upper": float(T[fit["end"]]),
                "rho0": fit["rho0"],
                "A": fit["A"],
                "reduced_chi2": fit["reduced_chi2"],
            }
        )

    return candidates


def extract_Tcoh_new(T, linecut, min_pts=5, min_T=1):
    """Find the largest extraction interval with at least 50% T² support."""
    T = np.asarray(T, float)
    min_T = np.ptp(T) * 0.2 if min_T is None else min_T

    # Reuse the local power-law fits so each temperature has an estimated
    # exponent n and its uncertainty.
    fit = extract_local_fits(T, linecut)
    n, n_sigma = np.asarray(fit["n"]), np.asarray(fit["n_sigma"])

    def get_T_idx(temp):
        return int(np.argmin(np.abs(T - temp)))

    def get_n_frac(idx_lower, idx_upper, n_lower, n_upper):
        # Convert each valid local fit into the probability that n lies within
        # the requested bounds.
        local_T = T[idx_lower:idx_upper]
        local_n = n[idx_lower:idx_upper]
        local_sigma = n_sigma[idx_lower:idx_upper]
        valid = np.isfinite(local_n) & np.isfinite(local_sigma) & (local_sigma > 0)
        if not np.any(valid):
            return 0.0

        local_T = local_T[valid]
        probabilities = norm.cdf(
            n_upper, loc=local_n[valid], scale=local_sigma[valid]
        ) - norm.cdf(n_lower, loc=local_n[valid], scale=local_sigma[valid])
        if len(local_T) == 1:
            return float(probabilities[0])

        # Weight by temperature span so dense low-T sampling does not dominate
        # wider, sparsely sampled high-T intervals.
        return float(np.trapezoid(probabilities, local_T) / np.ptp(local_T))

    # Use the existing extraction range, or the full measured range by default.
    extraction_range = next(
        (
            behavior
            for behavior in linecut.get("behaviors", [])
            if behavior.get("type") == "extraction_range"
        ),
        {"T_lower": T[0], "T_upper": T[-1]},
    )
    range_lower, range_upper = sorted(
        (extraction_range["T_lower"], extraction_range["T_upper"])
    )
    idx_lower = get_T_idx(range_lower)
    range_upper_idx = get_T_idx(range_upper)

    # Start with the smallest interval satisfying both minimum constraints.
    # idx_upper is exclusive throughout this function.
    idx_upper = max(
        idx_lower + min_pts,
        int(np.searchsorted(T, T[idx_lower] + min_T, side="left")) + 1,
    )

    if idx_upper > range_upper_idx + 1:
        return []
    if get_n_frac(idx_lower, idx_upper, 1.2, 8) < 0.5:
        return []

    # Grow the interval while its temperature-weighted quadratic support stays
    # above the acceptance threshold and within the extraction range.
    last_good_upper = None
    while idx_upper <= range_upper_idx + 1:
        if get_n_frac(idx_lower, idx_upper, 1.5, 2.5) < 0.5:
            break
        last_good_upper = idx_upper
        idx_upper += 1

    if last_good_upper is None:
        return []
    return [
        {
            "type": "extraction_Tcoh",
            "T_lower": T[idx_lower],
            "T_upper": T[last_good_upper - 1],
        }
    ]
