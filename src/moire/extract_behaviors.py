import numpy as np

from moire.extraction_helpers import adaptive_smooth, weighted_mad, weighted_median, smooth_mask, T_weights, moving_average

from hampel import hampel 
from scipy.signal import find_peaks
from scipy.stats import norm


# Extract candidate transition temperatures from sharp turns 
# Assumes T, rho are ordered and T is increasing
def extract_upturns(T, rho, threshold = 0.05) -> list[dict]:

    candidate_upturns = []

    # Smoothing + Deriverative Processing
    w = T_weights(T)
    rho_smoothed = adaptive_smooth(rho, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)
    d2pdT2_mov_avg = moving_average(d2pdT2, T)


    # Sharpness Metrics
    sharp = np.abs(d2pdT2)
    med_sharp = weighted_median(sharp, w)
    mad_sharp = weighted_mad(sharp, w)
    sharp_mov_avg = moving_average(sharp, T)


    # Find peaks
    peaks, prop = find_peaks(-rho_smoothed)
    rho_span = np.ptp(rho_smoothed)


    # Analyze peaks
    for idx in peaks:

        # Percentile of absolute value of 2nd deriverative at candidate assuming normal distribution 
        curvature_score = norm.cdf((sharp_mov_avg[idx] - med_sharp) / mad_sharp)

        # Finds features of cliff using by stopping when concavity changes
        idx_l = idx
        while idx_l > 0 and d2pdT2_mov_avg[idx_l] > 0:
            idx_l -= 1

        # Calculates cliff_size_score by comparing vertical size of cliff in range to half of total rho span
        cliff_size = abs(rho_smoothed[idx_l] - rho_smoothed[idx])
        cliff_size_score = 0 if (idx_l == 0 or idx_l == idx) else np.clip(2 * cliff_size / rho_span, 0, 1)

        # Skips if cliff size is too small
        if cliff_size_score < threshold: 
            continue

        comb_score = 0.2 * curvature_score + 0.8 * cliff_size_score
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


def extract_upturns_new(T, rho, min_feature_size = 0.5, sigma = 5, rho_noise = 20) -> list[dict]:

    # use interop for finding noise for rho at diff T ranges
    # or maybe just use measures - smoothed 
    
    # smooth data
    # find local minimums
    # for min in minimums:
        # score verical prominence compared to noise
        # -> 3 * sigma(noise) is decent prominence (get score of 0.5) (saturation func? 1 sig -> 0.2; 2 sig -> 0.4; 3sig -> 0.5)
        # score horizontal persistence compared to min_feature size 
        # -> min_feature_size gets 0.5 score (use 1st deg saturation func)
        # find final score by calculating geometric mean (sqrt(xy))
        # add to dict

    # TODO: in the future; do not do adaptive smooth (the rho passed in should be smoothed as part of the pipeline)
    candidate_upturns = []
    rho_smoothed = adaptive_smooth(rho, T)

    local_noise = rho_noise

    peaks, prop = find_peaks(-rho_smoothed, prominence = (None, None), height = (None, None))

    for i, idx in enumerate(peaks):
        
        # find vertical prominence -> score
        prominence = prop["prominences"][i]

        C_p = sigma / 4 # calibrates C_w such that # sigma of local noise gets 0.8 score 
        prom_z = (prominence / local_noise)
        prom_score = prom_z / (prom_z + C_p)

        # find horizontal persistence -> score 
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


    return candidate_upturns




# Extract candidate transition temperatures from change in curve fits (metallic transitions)
# Assumes T, rho are ordered and T is increasing
def extract_metallic_transitions(T, rho, candidates) -> list[dict]:
        
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

