import numpy as np 
from moire.extract_power_law import extract_local_fits
from moire.signal_helpers import moving_average

def get_fit_range(T, linecut, pos_frac=0.8) -> list[dict]:

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

    # Checking that > 80% of the range is positive
    if np.count_nonzero(dpdT[T_lower_idx : T_upper_idx + 1] > 0) / total_pts > pos_frac:
        behaviors = linecut.get("behaviors")
        behavior = {"type": "extraction_range", "T_lower": T_lower, "T_upper": T_upper}
        behaviors.append(behavior)

    return linecut


def extract_beheavior_fits(T, linecut, sub_threshold = 0.8, super_threshold = 1.2):

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

        n = extract_local_fits(T, linecut)["n"]
        n_avg = moving_average(n, T, 1)

        masks = {
            "sublinear": n_avg < sub_threshold,
            "linear": (sub_threshold <= n_avg) & (n_avg <= super_threshold),
            "superlinear": n_avg > super_threshold,
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
