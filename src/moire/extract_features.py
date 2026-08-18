import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2


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
    max_candidates=100,
    deviation=0.10,
    min_fit_points=6,
    min_fit_span=0.5,
    min_pvalue=0.05,
) -> list[dict]:
    """Return 10%-departure candidates from statistically compatible T² fits.

    ``min_pvalue`` is the minimum upper-tail chi-square GOF p-value accepted.
    Candidate confidence is this fit-compatibility p-value, not the probability
    that the reported temperature is the true Tcoh.
    """
    if not 0 <= min_pvalue <= 1:
        raise ValueError("min_pvalue must be between 0 and 1")
    if min_fit_points < 3:
        raise ValueError("min_fit_points must be at least 3")

    T = np.asarray(T, float)
    rho = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", rho), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float).copy()

    # Replace invalid noise estimates before the weighted T² fits.
    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    sigma[~valid_sigma] = np.median(sigma[valid_sigma]) if np.any(valid_sigma) else 1.0

    extraction_range = next(
        (
            behavior
            for behavior in linecut.get("behaviors", [])
            if behavior.get("type") == "extraction_range"
        ),
        {"T_lower": T[0], "T_upper": T[-1]},
    )
    lower, upper = sorted((extraction_range["T_lower"], extraction_range["T_upper"]))

    allowed = np.flatnonzero((T >= lower) & (T <= upper))
    if len(allowed) < min_fit_points + 1:
        return []

    first, last = allowed[0], allowed[-1]
    quadratic_fits = []

    # Fit expanding low-temperature windows to rho = rho0 + A*T². The final
    # allowed point is reserved so a departure can exist beyond the fit.
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
        chi_square = float(np.sum(residuals**2))
        degrees_of_freedom = len(residuals) - 2
        pvalue = float(chi2.sf(chi_square, degrees_of_freedom))
        if not np.isfinite(pvalue) or pvalue < min_pvalue:
            continue

        quadratic_fits.append(
            {
                "end": end,
                "rho0": float(rho0),
                "A": float(coefficient),
                "chi_square": chi_square,
                "degrees_of_freedom": degrees_of_freedom,
                "reduced_chi2": chi_square / degrees_of_freedom,
                "pvalue": pvalue,
            }
        )

    # Higher p-values mean the residual scatter is more compatible with the
    # supplied noise estimate. Prefer longer windows on exact ties.
    quadratic_fits.sort(key=lambda fit: (fit["pvalue"], T[fit["end"]] - T[first]), reverse=True)

    candidates = []
    for fit in quadratic_fits:
        # Apply the paper's 10% criterion directly to the smoothed linecut.
        predicted = fit["rho0"] + fit["A"] * T**2
        relative_difference = abs(smooth - predicted) / np.maximum(abs(predicted), 1e-12)
        departures = np.flatnonzero(relative_difference[fit["end"] + 1 : last + 1] >= deviation)
        if not len(departures):
            continue
        transition_idx = fit["end"] + 1 + departures[0]

        candidates.append(
            {
                "T": float(T[transition_idx]),
                "nu": linecut.get("nu"),
                "type": "Tcoh",
                "confidence": fit["pvalue"],
                "fit_T_lower": float(T[first]),
                "fit_T_upper": float(T[fit["end"]]),
                "rho0": fit["rho0"],
                "A": fit["A"],
                "chi_square": fit["chi_square"],
                "degrees_of_freedom": fit["degrees_of_freedom"],
                "reduced_chi2": fit["reduced_chi2"],
                "pvalue": fit["pvalue"],
            }
        )
        if len(candidates) == max_candidates:
            break

    return candidates
