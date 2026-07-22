import math
import numpy as np 

def update_scores_iter(linecuts, numIter, n_hood = 3):

    for i in range(numIter):
        new_score_name = f"score_{i}"
        if i == 0
            interp_score(linecuts, support_score="support")

    update_scores(linecuts, n_hood = n_hood)


def interp_score(linecuts, support_score = "support", old_score = "confidence", new_score = "score_1", c = 0.5):
    

    for i, linecut in enumerate(linecuts):
        
        features = linecut.get("features")

        for feat in features:

            # now interp score based on support scores 
            interp = feat.get(old_score) * c + feat.get(support_score) * (1-c)
            feat.update({new_score : interp})


def update_scores(linecuts, n_hood = 3):

    T = linecuts[0].get("T")

    # ---- NEW -----

    # iterate through each linecut
        # find the ones that contain features
        # for each feature:
            # find support score for left and right linecuts (check maybe 5 fillings across) based on score and T 
            # find max across left and right 
            # geometric mean between left and right
            # interp value 

    for i, linecut in enumerate(linecuts):

        # define left and right neighborhood
        # how to define boundary cases?
        features = linecut.get("features")
    
        if len(features) == 0:
            continue

        left_hood = linecuts[max(0, i-n_hood) : i]
        right_hood = linecuts[i+1 : min(len(linecuts), i+n_hood+1)]

        for feat in features: 

            T_feat = feat.get("T") 
            idx_feat = np.argmin(np.abs(T - T_feat))

            left_max_score = 0
            right_max_score = 0 

            for left_line in left_hood:

                left_features = left_line.get("features")
                
                for left_feat in left_features:

                    idx_left = np.argmin(np.abs(T - left_feat.get("T")))
                    conf = left_feat.get("confidence")
                    T_spacings = idx_feat - idx_left 

                    tau = 3 
                    score = conf * math.exp(-0.5*(T_spacings)**2 / tau)

                    if score > left_max_score:
                        left_max_score = score 

            for right_line in right_hood:

                right_features = right_line.get("features")

                for right_feat in right_features:
                    
                    idx_right = np.argmin(np.abs(T - right_feat.get("T")))
                    conf = right_feat.get("confidence")
                    T_spacings = idx_right - idx_feat 

                    tau = 3 
                    score = conf * math.exp(-0.5*(T_spacings)**2 / tau)

                    if score > right_max_score:
                        right_max_score = score 

            # Edge cases for boundaries 
            if 2 * len(left_hood) < n_hood and left_max_score == 0:
                left_max_score = right_max_score
            
            if 2 * len(right_hood) < n_hood and right_max_score == 0:
                right_max_score = left_max_score
                
            comb_support = left_max_score ** 0.5 * right_max_score ** 0.5
            feat.update({"support" : comb_support})
    

    for i, linecut in enumerate(linecuts):
        
        features = linecut.get("features")

        for feat in features:

            # now interp score based on support scores 
            c = 0.5 # how much of old score to retain 
            new_score = feat.get("confidence") * c + feat.get("support") * (1-c)
            feat.update({"score_1" : new_score})

    



    
    