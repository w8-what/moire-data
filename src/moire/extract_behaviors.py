import numpy as np
from moire.signal_helpers import adaptive_smooth, smooth_mask

from hampel import hampel 
from scipy.signal import find_peaks


def extract_upturns_new(T, linecut, min_feature_size = 0.5, sigma = 5) -> list[dict]:

    candidate_upturns = []
    rho_smoothed = linecut.get("rho_smoothed")
    local_noise = linecut.get("local_noise")

    peaks, prop = find_peaks(-rho_smoothed, prominence = (None, None), height = (None, None))

    for i, idx in enumerate(peaks):
        
        # Finding vertical prominence
        prominence = prop.get("prominences")[i]

        C_p = sigma / 4 # calibrates C_w such that # sigma of local noise gets 0.8 score 
        prom_z = (prominence / local_noise)
        prom_score = prom_z / (prom_z + C_p)

        # Finding horizontal persistence  
        left_base = prop["left_bases"][i]
        right_base = prop["right_bases"][i]
        width = T[right_base] - T[left_base]

        C_w = min_feature_size / 4 # calibrates C_w such that min_feature_size gets 0.8 score
        width_score = width / (width + C_w)

        comb_score = (prom_score * width_score) ** 0.5 
        comb_score = float(f"{comb_score:.3g}")

        candidate = {
            "T" : T[idx],
            "confidence" : comb_score,
            "phase_left" : "AFM",
            "phase_right" : None,
            "left_fit" : None,
            "right_fit" : None
        }
        
        candidate_upturns.append(candidate)


    # Add transition points if there are any peaks
    if len(candidate_upturns) != 0:
        candidate_upturns.insert(0,
            {
                "T" : T[0],
                "confidence" : 0.9,
                "phase_left" : None,
                "phase_right" : "AFM",
                "left_fit" : None,
                "right_fit" : None
            }
        )

    return candidate_upturns




# Extract candidate transition temperatures from change in curve fits (metallic transitions)
# Assumes T, rho are ordered and T is increasing
def extract_metallic_transitions(T, rho, candidates) -> list[dict]:

    # TODO: update with new linecut object
        
    T_left = 0 if len(candidates) == 0 else np.argmin(np.abs(T - candidates[-1].get("T"))) # Getting index of left most T
    rho_smoothed = adaptive_smooth(hampel(rho).filtered_data, T)
    dpdT = np.gradient(rho_smoothed, T) # Splice data to include right range only
    dpdT_pos = np.hstack((smooth_mask((dpdT > 0)[0:T_left]), smooth_mask((dpdT[T_left:] > 0)))) # Mask of spliced data (right side)

    i = T_left 
    curr_is_pos = dpdT_pos[i]
    phase_right = "Metal" if curr_is_pos else "Insulator"

    if T_left == 0:
        candidate = {
            "T" : T[0],
            "confidence" : 0.9,
            "phase_left" : None,
            "phase_right" : phase_right,
            "left_fit" : None,
            "right_fit" : None
        }
        candidates.append(candidate)
    else:
        candidates[-1].update({"phase_right" : phase_right})
    
    prev_is_pos = curr_is_pos
    i = i + 1 

    while i < len(T):
        curr_is_pos = dpdT_pos[i]
        if prev_is_pos != curr_is_pos:
            candidate = {
                "T" : T[i],
                "confidence" : 0.9,
                "phase_left" : "Insulator" if curr_is_pos else "Metal",
                "phase_right" : "Metal" if curr_is_pos else "Insulator",
                "left_fit" : None,
                "right_fit" : None
            }
            candidates.append(candidate)
        prev_is_pos = curr_is_pos
        i += 1

    candidate = {
        "T" : T[i-1],
        "confidence" : 0.9,
        "phase_left" : "Metal" if curr_is_pos else "Insulator",
        "phase_right" : None,
        "left_fit" : None,
        "right_fit" : None
    }
    candidates.append(candidate)
    return candidates

