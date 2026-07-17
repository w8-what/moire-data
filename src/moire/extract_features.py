import numpy as np
from scipy.signal import find_peaks


def extract_upturns(T, linecut, min_feature_size = 0.5, sigma = 5, coeff = 2) -> list[dict]:

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

        width = T[right_idx] - T[left_idx]

        C_w = min_feature_size**coeff / (0.8)/(1-0.8) # calibrates C_w such that min_feature_size gets 0.8 score
        width_score = width**coeff / (width**coeff + C_w)

        # Finding vertical prominence
        prominence = prop.get("prominences")[i]

        C_p = sigma**coeff * (0.8)/(1-0.8) # calibrates C_w such that # sigma ABOVE noise of local noise gets 0.8 score 

        local_noise = np.mean(noise[left_idx : right_idx + 1])
        prom_z = (prominence / local_noise)
        prom_score = prom_z**coeff / (prom_z**coeff + C_p)

        comb_score = (prom_score * width_score) ** 0.5 
        comb_score = float(f"{comb_score:.3g}")

        feature = {
            "T" : T[idx],
            "type" : "upturn",
            "confidence" : comb_score,
        }
        
        candidate_upturns.append(feature)

    return candidate_upturns

def extract_downturns(T, linecut, min_feature_size = 0.5, sigma = 5, coeff = 2) -> list[dict]:

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

                j += 1

            left_idx = j
            right_idx = right_base_idx

        else:
            # use left as point and find right point that corresponds to right
            j = idx + 1 # watch out edge 
            while j >= left_base_idx:
                low = min(rho_smoothed[j - 1], rho_smoothed[j])
                high = max(rho_smoothed[j - 1], rho_smoothed[j])

                if low <= rho_horizontal and rho_horizontal <= high:
                    break

                j -= 1

            left_idx = left_base_idx
            right_idx = j

        width = T[right_idx] - T[left_idx]

        C_w = min_feature_size**coeff / (0.8)/(1-0.8) # calibrates C_w such that min_feature_size gets 0.8 score
        width_score = width**coeff / (width**coeff + C_w)

        # Finding vertical prominence
        prominence = prop.get("prominences")[i]

        C_p = sigma**coeff * (0.8)/(1-0.8) # calibrates C_w such that # sigma ABOVE noise of local noise gets 0.8 score 

        local_noise = np.mean(noise[left_idx : right_idx + 1])
        prom_z = (prominence / local_noise)
        prom_score = prom_z**coeff / (prom_z**coeff + C_p)

        comb_score = (prom_score * width_score) ** 0.5 
        comb_score = float(f"{comb_score:.3g}")


        feature = {
            "T" : T[idx],
            "type" : "downturn",
            "confidence" : comb_score,
        }
        
        candidate_downturns.append(feature)

    return candidate_downturns