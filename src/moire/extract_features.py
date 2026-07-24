import numpy as np
from scipy.signal import find_peaks

def _hill_sigmoid(x, reference_value, reference_score = 0.8, coeff = 2):

    C = reference_value**coeff * (1 - reference_score) / reference_score
    return x**coeff / (x**coeff + C)



def extract_upturns(T, linecut, min_pts = 5, min_width = 0.5, sigma = 5, coeff = 2) -> list[dict]:

    candidate_upturns = []
    rho_smoothed = linecut.get("rho_smoothed")
    noise = linecut.get("local_noise")

    peaks, prop = find_peaks(-rho_smoothed, prominence = (None, None), height = (None, None))

    for i, idx in enumerate(peaks):
        
        # Finding horizontal persistence
        left_base_idx = prop["left_bases"][i]
        right_base_idx = prop["right_bases"][i]

        rho_horizontal = min(rho_smoothed[right_base_idx], rho_smoothed[left_base_idx])

        if rho_smoothed[right_base_idx] - rho_smoothed[left_base_idx] > 0:
            # use right as point and find right point that corresponds to right
            j = idx + 1 # watch out edge 
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
            j = idx - 1 # watch out edge 
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
        prom_z = (prominence / local_noise)

        width = T[right_idx] - T[left_idx]
        pts = len(T[left_idx:right_idx+1])

        target = 0.8

        C_prom = sigma**coeff * (1 - target) / target
        C_width = min_width**coeff * (1 - target) / target
        C_pts = min_pts**coeff * (1 - target) / target

        prom_score = prom_z**coeff / (prom_z**coeff + C_prom)
        pts_score = pts**coeff / (pts**coeff + C_pts)
        width_score = width**coeff / (width**coeff + C_width)

        comb_score = prom_score ** 0.5 * pts_score ** 0.3 * width_score ** 0.2 
        comb_score = float(f"{comb_score:.3g}")

        feature = {
            "T" : T[idx],
            "nu" : linecut.get("nu"),
            "type" : "upturn",
            "confidence" : comb_score,
        }
        
        candidate_upturns.append(feature)

    return candidate_upturns

def extract_downturns(T, linecut, min_pts = 5, min_width = 0.5, sigma = 5, coeff = 2) -> list[dict]:

    candidate_downturns = []
    rho_smoothed = linecut.get("rho_smoothed")
    noise = linecut.get("local_noise")

    peaks, prop = find_peaks(rho_smoothed, prominence = (None, None), height = (None, None))

    for i, idx in enumerate(peaks):
        
        # Finding horizontal persistence
        left_base_idx = prop["left_bases"][i]
        right_base_idx = prop["right_bases"][i]

        rho_horizontal = max(rho_smoothed[right_base_idx], rho_smoothed[left_base_idx])

        if rho_smoothed[right_base_idx] - rho_smoothed[left_base_idx] > 0:
            # use right as point and find left point that corresponds to right
            j = idx - 1 # watch out edge 
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
            j = idx + 1 # watch out edge 
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
        prom_z = (prominence / local_noise)

        width = T[right_idx] - T[left_idx]
        pts = len(T[left_idx:right_idx+1])

        target = 0.8

        C_prom = sigma**coeff * (1 - target) / target
        C_width = min_width**coeff * (1 - target) / target
        C_pts = min_pts**coeff * (1 - target) / target

        prom_score = prom_z**coeff / (prom_z**coeff + C_prom)
        pts_score = pts**coeff / (pts**coeff + C_pts)
        width_score = width**coeff / (width**coeff + C_width)

        comb_score = prom_score ** 0.5 * pts_score ** 0.3 * width_score ** 0.2 
        comb_score = float(f"{comb_score:.3g}")

        feature = {
            "T" : T[idx],
            "nu" : linecut.get("nu"),
            "type" : "downturn",
            "confidence" : comb_score,
        }
        
        candidate_downturns.append(feature)

    return candidate_downturns



def extract_Tc(T, linecut, threshold = 20):

    # find each point that are below the resistivity threshold
    # for each point calculate the following
        # 1. the temp fraction that is below the resistivity threshold
        # 2. the number of points under the temperature
        # 3. the temperature range of the threshold
        # 4. use geometric mean for scoring

    candidate_Tcs = []

    rho = linecut.get("rho")
    below = rho <= threshold
    candidates_idx = np.flatnonzero(below)

    for candidate in candidates:
        num_points = np.argmin(np.abs(T - candidate))
        temp_range = candidate

        T_lower = T[:num_points + 1]
        below_lower = rho[:num_points + 1] < threshold

        temp_frac = np.trapezoid(
            below_lower.astype(float),
            T_lower,
        ) / (T_lower[-1] - T_lower[0])

        score_pts = _hill_sigmoid(num_points, 5)
        score_temp = _hill_sigmoid(temp_range, 0.5)
        score_frac = _hill_sigmoid(temp_frac, 0.9)

        comb_score = score_frac**(1/3) * score_pts**(1/3) * score_temp**(1/3)

        feature = {
            "T" : candidate,
            "nu" : linecut.get("nu"),
            "type" : "T_c",
            "confidence" : comb_score,
        } 

        candidate_Tcs.append(feature)

    candidate_Tcs.sort(key=lambda feature: feature["confidence"], reverse=True)

    return candidate_Tcs[:3]
