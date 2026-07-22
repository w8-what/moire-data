import math

def update_scoring(linecuts, n_hood = 3):

    T = linecuts.get(T)

    # ---- NEW -----
    # iterate through each linecut
    # find support score for left and right linecuts (check maybe 5 fillings across)
    # find max across left and right 
    # geometric mean between left and right
    # interp value 

    support = []

    for i, linecut in enumerate(linecuts):

        # define left and right neighborhood
        # how to define boundary cases?
        left_hood = linecuts[max(0, i-n_hood) : i]
        right_hood = linecuts[i+1 : min(len(linecuts), i+n_hood+1)]

        left_max_score = 0
        right_max_score = 0 

        for left_line in left_hood:
            # see if there is feature
            # if feature -> calculate score and get score
            features = left_line.get("features")
            
            for feat in features:
                conf = feat.get("confidence")
                T_spacings = 
                tau = 
                score = conf * math.exp(-0.5*(T_spacings)**2 / tau)

                if score > left_max_score:
                    left_max_score = score 

        for right_line in right_hood:
            
            features = right_line.get("features")
            
            for feat in features:
                conf = feat.get("confidence")
                T_spacings = 
                tau = 
                score = conf * math.exp(-0.5*(T_spacings)**2 / tau)

                if score > left_max_score:
                    left_max_score = score 

            
        # update left to match right; or right to match left score IF uh two scenarios
            # the left_hood is less than half of the hood requirement or right_hood is less than half of the hood req
            # and also there were no candidates found 

        if 2 * len(left_hood) < n_hood:
            if left_max_score == 0:
                left_max_score = right_max_score
        
        if 2 * len(right_hood) < n_hood:
            if right_max_score == 0:
                right_max_score = left_max_score
            
        total_support = left_max_score ** 0.5 * right_max_score ** 0.5
        support.append(total_support)
    
    for i, linecut in enumerate(linecuts):

        # now interp score based on support scores 
        c = 0.5 # how much of old score to retain 
        new_score = linecut.get("score") * c + support[i] * (1-c)
        linecut.update({"new_score_1" : new_score})



    
    