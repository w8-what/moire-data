import numpy as np 
from moire.extract_power_law import extract_local_fits
from moire.signal_helpers import moving_average
from copy import deepcopy

def _hill_sigmoid(x, reference_value, reference_score=0.8, coeff=2):

    C = reference_value**coeff * (1 - reference_score) / reference_score
    return x**coeff / (x**coeff + C)

def extract_fit_range(T, linecut, pos_frac=0.8) -> list[dict]:

    T_lower = T[0]
    T_upper = T[-1]
    features = linecut.get("features_new")
    rho_smoothed = linecut.get("rho_smoothed")

    # Updating the lower bound to be the highest upturn
    for feat in features:
        if feat.get("type") == "upturn":
            T_feature = feat.get("T")
            if T_feature > T_lower:
                T_lower = T_feature

    # Updating the upper bound to be the lowest downturn if downturn exists in current bound
    for feat in features:
        if feat.get("type") == "downturn":
            T_feature = feat.get("T")
            if T_lower < T_feature and T_feature < T_upper:
                T_upper = T_feature

    # Updating the lower bound to be highest Tc if Tc exists in current bound
    for feat in features:
        if feat.get("type") == "Tc":
            T_feature = feat.get("T")
            if T_lower < T_feature and T_feature < T_upper:
                if T_feature > T_lower:
                    T_lower = T_feature

    T_lower_idx = np.argmin(np.abs(T - T_lower))
    T_upper_idx = np.argmin(np.abs(T - T_upper))
    dpdT = np.gradient(rho_smoothed, T)
    total_pts = T_upper_idx - T_lower_idx

    behavior = []
    # Checking that > 80% of the range is positive
    if np.count_nonzero(dpdT[T_lower_idx : T_upper_idx + 1] > 0) / total_pts > pos_frac:
        behavior.append({"type": "extraction_range", "T_lower": T_lower, "T_upper": T_upper})

    return behavior


def extract_behavior_fits(T, linecut, sub_threshold = 0.8, super_threshold = 1.2):

    # find regions of linear, superlinear, and sublinear
    # takes into account of uncertainity somehow 

    # find masks of lienar, superlinear, and sublinear
    # label them rudimentarily as linear superlinear, / sublinear
    # see phase diagram


    # potential redo:
        # label linear, superlinear, sublinear
        # additionally superlinear / sublinear as linear-compatiable 
        # and additinally linear as either yea yeah you get the point
        # and then somehow you can find a global interpretation that uses agreement?
        # or perhaps a linecut interpretation not sure
        # well lets see how the sigma is even calculated? 
        # itneresting problem actually kind of depends on the uncertainity in N and then we can kind of
        # go from there ideally grow using like a diffuser that diffuses from high conf -> low conf 
        # however yeah there are not a lot of noises it seems like

        behaviors = []

        n = np.asarray(extract_local_fits(T, linecut)["n"])

        # Taking the moving average within fit_range
        for behavior in linecut["behaviors"]:
            if behavior["type"] == "extraction_range":
                left_idx = np.argmin(np.abs(T - behavior["T_lower"]))
                right_idx = np.argmin(np.abs(T - behavior["T_upper"]))
                s = slice(left_idx,right_idx+1)
                n[s] = moving_average(n[s], T[s], 1)

        masks = {
            "sublinear": n < sub_threshold,
            "linear": (sub_threshold <= n) & (n <= super_threshold),
            "superlinear": n > super_threshold,
        }

        for behavior_type, mask in masks.items():
            # Find contiguous True intervals using [start, end).
            changes = np.diff(np.r_[False, mask, False].astype(int))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)

            for start, end in zip(starts, ends):
                # Extend each endpoint halfway toward the neighboring T point.
                if start > 0:
                    T_lower = (T[start - 1] + T[start]) / 2
                else:
                    T_lower = T[0] - (T[1] - T[0]) / 2

                if end < len(T):
                    T_upper = (T[end - 1] + T[end]) / 2
                else:
                    T_upper = T[-1] + (T[-1] - T[-2]) / 2

                behaviors.append({
                    "type": behavior_type,
                    "T_lower": T_lower,
                    "T_upper": T_upper,
                })

        return behaviors


def refine_behaviors(T, linecut, min_points=1, min_T=0):
    # first pass: remove tiny islands (either 1 point, or T range less than some threshold)
    # second pass: ???


    behaviors = linecut["behaviors"]
    refined_behaviors = sorted(
        [behavior.copy() for behavior in behaviors if behavior["type"] != "extraction_range"],
        key = lambda b:b["T_lower"]
    )

    # so what we do?
    # find the behaviors that are less than T range or 1 point large 
    # and then update them based on surrounding linecuts 
    # if there is disagreement, then use whichone is largest in confidence? yeah uh i dont really have a confidence score thats the thing uh
    # hmm well lets assume we have that
    # OK and so we do that to refine it

    def get_T_idx(temp):
        return np.argmin(np.abs(T - temp))
    
    for i, behavior in enumerate(refined_behaviors):

        T_upper = behavior["T_upper"]
        T_lower = behavior["T_lower"]
        T_range = T_upper - T_lower
    
        num_pts = get_T_idx(T_upper) - get_T_idx(T_lower) + 1

        if num_pts <= min_points or T_range <= min_T:

            # perform the thing 
            # look at top and bottom and see if can reconcile
            # if not then uh label as unclassified lol

            if i == 0:
                behavior["type"] = refined_behaviors[i+1]["type"]
            elif i == len(refined_behaviors) - 1:
                behavior["type"] = refined_behaviors[i-1]["type"]
            else: 
                behavior_upper = refined_behaviors[i+1]["type"]
                behavior_lower = refined_behaviors[i-1]["type"]
                if behavior_upper == behavior_lower:
                    behavior["type"] = behavior_upper 
                else: 
                    behavior["type"] = "unlabeled"


    return refined_behaviors