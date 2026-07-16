import numpy as np
from moire.signal_helpers import adaptive_smooth, smooth_mask

from hampel import hampel 
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


        candidate = {
            "T" : T[idx],
            "type" : "upturn",
            "confidence" : comb_score,
            "phase_left" : "AFM",
            "phase_right" : None,
            "left_fit" : None,
            "right_fit" : None
        }
        
        candidate_upturns.append(candidate)


    # # Add transition points if there are any peaks
    # if len(candidate_upturns) != 0:
    #     candidate_upturns.insert(0,
    #         {
    #             "T" : T[0],
    #             "confidence" : 0.9,
    #             "phase_left" : None,
    #             "phase_right" : "AFM",
    #             "left_fit" : None,
    #             "right_fit" : None
    #         }
    #     )

    return candidate_upturns






def extract_downturns(T, linecut, min_feature_size = 0.5, sigma = 5, coeff = 2) -> list[dict]:

    candidate_upturns = []
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


        candidate = {
            "T" : T[idx],
            "type" : "downturn", 
            "confidence" : comb_score,
            "phase_left" : "Metal",
            "phase_right" : "Insulator",
            "left_fit" : None,
            "right_fit" : None
        }
        
        candidate_upturns.append(candidate)

    return candidate_upturns



# # Extract candidate transition temperatures from change in curve fits (metallic transitions)
# # Assumes T, rho are ordered and T is increasing
# def extract_downturns(T, rho, candidates) -> list[dict]:

#     # TODO: update with new linecut object
        
#     T_left = 0 if len(candidates) == 0 else np.argmin(np.abs(T - candidates[-1].get("T"))) # Getting index of left most T
#     rho_smoothed = adaptive_smooth(hampel(rho).filtered_data, T)
#     dpdT = np.gradient(rho_smoothed, T) # Splice data to include right range only
#     dpdT_pos = np.hstack((smooth_mask((dpdT > 0)[0:T_left]), smooth_mask((dpdT[T_left:] > 0)))) # Mask of spliced data (right side)

#     i = T_left 
#     curr_is_pos = dpdT_pos[i]
#     phase_right = "Metal" if curr_is_pos else "Insulator"

#     if T_left == 0:
#         candidate = {
#             "T" : T[0],
#             "confidence" : 0.9,
#             "phase_left" : None,
#             "phase_right" : phase_right,
#             "left_fit" : None,
#             "right_fit" : None
#         }
#         candidates.append(candidate)
#     else:
#         candidates[-1].update({"phase_right" : phase_right})
    
#     prev_is_pos = curr_is_pos
#     i = i + 1 

#     while i < len(T):
#         curr_is_pos = dpdT_pos[i]
#         if prev_is_pos != curr_is_pos:
#             candidate = {
#                 "T" : T[i],
#                 "confidence" : 0.9,
#                 "phase_left" : "Insulator" if curr_is_pos else "Metal",
#                 "phase_right" : "Metal" if curr_is_pos else "Insulator",
#                 "left_fit" : None,
#                 "right_fit" : None
#             }
#             candidates.append(candidate)
#         prev_is_pos = curr_is_pos
#         i += 1

#     candidate = {
#         "T" : T[i-1],
#         "confidence" : 0.9,
#         "phase_left" : "Metal" if curr_is_pos else "Insulator",
#         "phase_right" : None,
#         "left_fit" : None,
#         "right_fit" : None
#     }
#     candidates.append(candidate)
#     return candidates

