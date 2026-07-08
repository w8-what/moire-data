from moire.extraction_helpers import *

from scipy.signal import find_peaks
from scipy.stats import norm

# Extract candidate transition temperatures from sharp turns 
def extract_upturns(T, rho, sensitivity = 1) -> list[dict]:

    candidate_upturns = []
    threshold = sensitivity * 100

    # rho = hampel(rho).filtered_data
    rho_smoothed = adaptive_smooth(rho, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)

    peaks, prop = find_peaks(-rho_smoothed)

    dpdT_neg = dpdT[np.where(dpdT < 0)] # array of all negative dpdT values
    med_dpdT_neg = np.median(dpdT_neg)
    mad_dpdT_neg = mad(dpdT_neg)

    med_d2pdT2 = np.median(d2pdT2)
    mad_d2pdT2 = mad(d2pdT2)

    kernel = np.ones(3)/3
    d2pdT2_moving_average = np.convolve(d2pdT2, kernel, mode = "same")
    rho_span = np.max(rho_smoothed) - np.min(rho_smoothed)

    if len(peaks) != 0:
        candidate_upturns.append(
            {
                "T" : T[0],
                "confidence" : 0.9,
                "phase_left" : None,
                "phase_right" : "AFM",
                "left_fit" : None,
                "right_fit" : None
            }
        )

    for idx in peaks:

        # Percentile of second deriverative at candidate assuming normal distribution 
        curvature_score = norm.cdf((d2pdT2[idx] - med_d2pdT2) / mad_d2pdT2)

        # Finds features of cliff using by stopping when concavity changes
        idx_l = idx
        while idx_l > 0 and d2pdT2_moving_average[idx_l] > 0:
            idx_l -= 1

        # Calculates cliff_slope_score by comparing average slope in range to median using MAD
        avg_dpdT_cliff = 0 if (idx_l == 0 or idx_l == idx) else np.mean(dpdT[idx_l:idx])
        cliff_slope_score = norm.cdf(-(avg_dpdT_cliff - med_dpdT_neg) / mad_dpdT_neg) # more negative z-score is desirable here (steeper slope)

        # Calculates cliff_size_score by comparing vertical size of cliff in range to half of total rho span
        cliff_size = abs(rho_smoothed[idx_l] - rho_smoothed[idx])
        cliff_size_score = 0 if (idx_l == 0 or idx_l == idx) else 2 * cliff_size / rho_span

        # print(f"{idx_l=}")
        # print(f"{T[idx_l]=}")
        # print(f"{curvature_score=}")
        # print(f"{cliff_slope_score=}")
        # print(f"{cliff_size_score=}\n")

        # print(f"{(d2pdT2_moving_average[idx_l] - med_d2pdT2) / mad_d2pdT2 =}")


        comb_score = 0.2 * curvature_score + 0.2 * cliff_slope_score + 0.6 * cliff_size_score
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

