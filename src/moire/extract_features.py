import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import chi2, norm


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


def _fit_power_law_exponent(T, rho, sigma, exponent_bounds):
    """Robustly fit rho = rho0 + A*T**n and estimate uncertainty in n."""
    T_reference = float(np.median(T))
    if not np.isfinite(T_reference) or T_reference <= 0:
        return None

    scaled_T = T / T_reference
    design = np.column_stack((np.ones_like(scaled_T), scaled_T))
    rho0, coefficient = np.linalg.lstsq(design, rho, rcond=None)[0]
    coefficient = max(float(coefficient), np.finfo(float).eps)

    def residuals(parameters):
        offset, scale, exponent = parameters
        return (offset + scale * scaled_T**exponent - rho) / sigma

    result = least_squares(
        residuals,
        [rho0, coefficient, 0.8],
        bounds=([-np.inf, 0.0, exponent_bounds[0]], [np.inf, np.inf, exponent_bounds[1]]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None

    offset, scale, exponent = (float(value) for value in result.x)
    exponent_sigma = np.nan
    degrees_of_freedom = len(T) - 3
    normal_matrix = result.jac.T @ result.jac
    if degrees_of_freedom > 0 and np.linalg.matrix_rank(normal_matrix) == 3:
        covariance = np.linalg.inv(normal_matrix) * (2 * result.cost / degrees_of_freedom)
        exponent_variance = float(covariance[2, 2])
        if np.isfinite(exponent_variance) and exponent_variance >= 0:
            exponent_sigma = float(np.sqrt(exponent_variance))

    if np.isclose(exponent, exponent_bounds, rtol=0, atol=1e-4).any():
        exponent_sigma = np.nan

    return {
        "rho0": offset,
        "A": scale / T_reference**exponent,
        "n": exponent,
        "n_sigma": exponent_sigma,
    }


def _persistent_lower_departure(
    T,
    relative_difference,
    absolute_difference,
    sigma,
    transition_idx,
    first,
    deviation,
    min_points,
    min_span,
    min_fraction,
    noise_threshold,
):
    """Check that a downward 10% crossing persists toward lower temperature."""
    lower_idx = transition_idx
    while lower_idx >= first:
        point_count = transition_idx - lower_idx + 1
        span = T[transition_idx] - T[lower_idx]
        if point_count >= min_points and span >= min_span:
            selection = slice(lower_idx, transition_idx + 1)
            departure_fraction = float(np.mean(relative_difference[selection] >= deviation))
            noise_fraction = float(
                np.mean(absolute_difference[selection] >= noise_threshold * sigma[selection])
            )
            if departure_fraction >= min_fraction and noise_fraction >= min_fraction:
                return {
                    "T_lower": float(T[lower_idx]),
                    "T_upper": float(T[transition_idx]),
                    "points": point_count,
                    "span": float(span),
                    "departure_fraction": departure_fraction,
                    "noise_fraction": noise_fraction,
                }
            return None
        lower_idx -= 1

    return None


def _fit_lower_power_law(
    T, rho, sigma, transition_idx, first, min_points, min_span, exponent_bounds
):
    """Fit the nearest identifiable power-law window below a departure."""
    lower_idx = transition_idx
    while lower_idx >= first:
        point_count = transition_idx - lower_idx + 1
        span = T[transition_idx] - T[lower_idx]
        if point_count >= min_points and span >= min_span:
            selection = slice(lower_idx, transition_idx + 1)
            fit = _fit_power_law_exponent(
                T[selection], rho[selection], sigma[selection], exponent_bounds
            )
            if fit is not None and np.isfinite(fit["n_sigma"]) and fit["n_sigma"] >= 0:
                return fit, lower_idx
        lower_idx -= 1

    return None


def extract_Tprime(
    T,
    linecut,
    max_candidates=100,
    deviation=0.10,
    min_fit_points=6,
    min_fit_span=0.5,
    min_pvalue=0.05,
    persistence_points=4,
    persistence_span=0.3,
    persistence_fraction=0.8,
    noise_threshold=1.5,
    min_sublinear_points=6,
    min_sublinear_span=0.5,
    min_sublinear_probability=0.5,
    exponent_bounds=(0.1, 4.0),
) -> list[dict]:
    """Return paper-defined T-prime candidates from high-T linear fits.

    Each fit is anchored to the top of the extraction range. The candidate is
    the first persistent 10% departure found while scanning downward in
    temperature. Candidates are retained only when the lower-temperature side
    is statistically compatible with a sublinear power law (n < 1).

    ``confidence`` is the upper-tail chi-square p-value of the high-temperature
    linear fit. It is a fit-compatibility score, not the probability that the
    reported temperature is the true T-prime.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if not 0 < deviation < 1:
        raise ValueError("deviation must be between 0 and 1")
    if min_fit_points < 3:
        raise ValueError("min_fit_points must be at least 3")
    if min_fit_span <= 0:
        raise ValueError("min_fit_span must be positive")
    if not 0 <= min_pvalue <= 1:
        raise ValueError("min_pvalue must be between 0 and 1")
    if persistence_points < 1:
        raise ValueError("persistence_points must be at least 1")
    if persistence_span < 0:
        raise ValueError("persistence_span cannot be negative")
    if not 0 < persistence_fraction <= 1:
        raise ValueError("persistence_fraction must be between 0 and 1")
    if noise_threshold < 0:
        raise ValueError("noise_threshold cannot be negative")
    if min_sublinear_points < 4:
        raise ValueError("min_sublinear_points must be at least 4")
    if min_sublinear_span <= 0:
        raise ValueError("min_sublinear_span must be positive")
    if not 0 <= min_sublinear_probability <= 1:
        raise ValueError("min_sublinear_probability must be between 0 and 1")
    if len(exponent_bounds) != 2 or not 0 < exponent_bounds[0] < 1 < exponent_bounds[1]:
        raise ValueError("exponent_bounds must straddle 1 and contain positive values")

    T = np.asarray(T, float)
    rho = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", rho), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float).copy()
    if not (T.ndim == rho.ndim == smooth.ndim == sigma.ndim == 1):
        raise ValueError("T, rho, rho_smoothed, and local_noise must be one-dimensional")
    if not (len(T) == len(rho) == len(smooth) == len(sigma)):
        raise ValueError("T and linecut arrays must have equal length")
    if not len(T):
        return []
    if not np.all(np.isfinite(T)) or not np.all(T > 0) or not np.all(np.diff(T) > 0):
        raise ValueError("T must be finite, positive, and strictly increasing")
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(smooth)):
        raise ValueError("rho and rho_smoothed must contain only finite values")

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
    required_points = max(min_fit_points + 1, min_sublinear_points + 1)
    if len(allowed) < required_points:
        return []

    first, last = int(allowed[0]), int(allowed[-1])
    linear_fits = []

    # Fit shrinking high-temperature windows to rho = rho0 + A*T. At least
    # one lower-temperature point is reserved for finding a departure.
    for start in allowed[1:]:
        if last - start + 1 < min_fit_points:
            break
        if T[last] - T[start] < min_fit_span:
            continue

        selection = slice(start, last + 1)
        design = np.column_stack((np.ones(last - start + 1), T[selection]))
        weighted_design = design / sigma[selection, None]
        rho0, coefficient = np.linalg.lstsq(
            weighted_design, rho[selection] / sigma[selection], rcond=None
        )[0]
        if coefficient <= 0:
            continue

        fitted = rho0 + coefficient * T[selection]
        residuals = (rho[selection] - fitted) / sigma[selection]
        chi_square = float(np.sum(residuals**2))
        degrees_of_freedom = len(residuals) - 2
        pvalue = float(chi2.sf(chi_square, degrees_of_freedom))
        if not np.isfinite(pvalue) or pvalue < min_pvalue:
            continue

        linear_fits.append(
            {
                "start": int(start),
                "rho0": float(rho0),
                "A": float(coefficient),
                "chi_square": chi_square,
                "degrees_of_freedom": degrees_of_freedom,
                "reduced_chi2": chi_square / degrees_of_freedom,
                "pvalue": pvalue,
            }
        )

    linear_fits.sort(key=lambda fit: (fit["pvalue"], T[last] - T[fit["start"]]), reverse=True)

    candidates = []
    for fit in linear_fits:
        predicted = fit["rho0"] + fit["A"] * T
        absolute_difference = np.abs(smooth - predicted)
        relative_difference = absolute_difference / np.maximum(np.abs(predicted), 1e-12)

        for transition_idx in range(fit["start"] - 1, first - 1, -1):
            if relative_difference[transition_idx] < deviation:
                continue

            persistence = _persistent_lower_departure(
                T,
                relative_difference,
                absolute_difference,
                sigma,
                transition_idx,
                first,
                deviation,
                persistence_points,
                persistence_span,
                persistence_fraction,
                noise_threshold,
            )
            if persistence is None:
                continue

            lower_fit = _fit_lower_power_law(
                T,
                rho,
                sigma,
                transition_idx,
                first,
                min_sublinear_points,
                min_sublinear_span,
                exponent_bounds,
            )
            if lower_fit is None:
                continue
            sublinear_fit, sublinear_lower_idx = lower_fit
            exponent = sublinear_fit["n"]
            exponent_sigma = sublinear_fit["n_sigma"]
            if exponent_sigma == 0:
                sublinear_probability = float(exponent < 1)
            else:
                sublinear_probability = float(norm.cdf((1 - exponent) / exponent_sigma))
            if sublinear_probability < min_sublinear_probability:
                continue

            candidates.append(
                {
                    "T": float(T[transition_idx]),
                    "nu": linecut.get("nu"),
                    "type": "Tprime",
                    "confidence": fit["pvalue"],
                    "fit_T_lower": float(T[fit["start"]]),
                    "fit_T_upper": float(T[last]),
                    "rho0": fit["rho0"],
                    "A": fit["A"],
                    "chi_square": fit["chi_square"],
                    "degrees_of_freedom": fit["degrees_of_freedom"],
                    "reduced_chi2": fit["reduced_chi2"],
                    "pvalue": fit["pvalue"],
                    "persistence_T_lower": persistence["T_lower"],
                    "persistence_T_upper": persistence["T_upper"],
                    "persistence_points": persistence["points"],
                    "persistence_span": persistence["span"],
                    "persistence_fraction": persistence["departure_fraction"],
                    "noise_fraction": persistence["noise_fraction"],
                    "sublinear_fit_T_lower": float(T[sublinear_lower_idx]),
                    "sublinear_fit_T_upper": float(T[transition_idx]),
                    "sublinear_rho0": sublinear_fit["rho0"],
                    "sublinear_A": sublinear_fit["A"],
                    "sublinear_n": exponent,
                    "sublinear_n_sigma": exponent_sigma,
                    "sublinear_probability": sublinear_probability,
                }
            )
            break

        if len(candidates) == max_candidates:
            break

    return candidates
