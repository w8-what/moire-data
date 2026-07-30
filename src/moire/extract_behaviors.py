import numpy as np 

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


def extract_beheavior_fits(T, linecut, max_candidates=3):

    # find regions of linear, superlinear, and sublinear
    # takes into account of uncertainity somehow 

    # find masks of lienar, superlinear, and sublinear
    # additionally find masks of 

    return None