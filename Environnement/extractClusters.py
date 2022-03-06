import numpy as np
from sklearn.metrics import hamming_loss
import time 
from tqdm import trange

def score_fn(preds, trues):
    """
    Compute the closeness between two lists of strings

    For each category, we add the hamming distance to the overall score.
    """
    preds = np.array(preds)
    trues = np.array(trues)
    score = 0
    for i in range(len(preds)):
        n,m = len(preds[i]), len(trues[i])
        if m==0 and n==0:
            # Both are empty
            # score += 0
            continue

        elif n==0:
            # Pred is empty
            # score is equal to one
            score +=1
            continue

        elif m==0:
            # 
            score +=1
            continue

        elif n>m:
            # Pred is longer than trues
            # We take the first m elements because after it is
            # necessarily different
            preds[i] = preds[i][:m]
 
        elif m>n:
            # Trues is longer than preds
            # We take the first m elements because after it is
            # necessarily different
            trues[i] = trues[i][:n]

        # We compute the hamming distance
        # the first term is the number of different elements
        # the second term is the difference in length
        score += (np.sum(np.array(list(preds[i])) == np.array(list(trues[i])))+ np.abs(n-m))/max(n,m)
    return score



def get_closest_keys_location(key, dfs):

    """
    Return the closest keys to the given key using a scoring function.
    """

    # We use these dataframes because they reduce the computation time
    # by reducing the number of comparisons
    start = time.time()
    cols = ["State", "District", "SubDistrict", "Block", "GP"]

    state, district, subDistrict, block, GP = key.split("_")[:5]
    df_State = dfs[state]

    df_District = df_State[df_State["District"] == district].copy()
    df_SubDistrict = df_District[df_District["SubDistrict"] == subDistrict].copy()
    df_Block = df_SubDistrict[df_SubDistrict["Block"] == block].copy()
    df_GP = df_Block[df_Block["GP"] == GP].copy()
    # print("Time to precompute the dataframes: ", time.time()-start)

    if len(df_GP)>0:

        return df_GP

    elif len(df_Block)>0:
        return df_Block

    elif len(df_SubDistrict)>0:
        return df_SubDistrict

    elif len(df_District)>0:
        return df_District

    return df_State


def get_closest_keys_scoring(key, dfs, score_fn=score_fn):

    """
    Return the closest keys to the given key using a scoring function.
    """

    # We use these dataframes because they reduce the computation time
    # by reducing the number of comparisons
    start = time.time()
    cols = ["State", "District", "SubDistrict", "Block", "GP"]
    state, district, subDistrict, block, GP = key.split("_")[:5]
    df_State = dfs[state]
    df_District = df_State[df_State["District"] == district].copy()
    df_SubDistrict = df_District[df_District["SubDistrict"] == subDistrict].copy()
    df_Block = df_SubDistrict[df_SubDistrict["Block"] == block].copy()
    df_GP = df_Block[df_Block["GP"] == GP].copy()
    # print("Time to precompute the dataframes: ", time.time()-start)

    start = time.time()
    if len(df_GP)>0:
        # In this case, we have exact matches
        # No need to use the scoring function
        # print("Time to answer the query with GP: ", time.time()-start)

        return df_GP

    elif len(df_Block)>0:
        # Faster to use the scoring function on this dataframe than on the whole df or the next dataframes
        df_Block["score"] = df_Block.apply(lambda x: score_fn(x[cols], 
                                        np.array([state, district, subDistrict, block, GP])), axis=1)
        # Return the closest keys in the state regarding score_fn
        best = df_Block.sort_values(by=["score"], ascending=False).iloc[0]["score"]
        # print("Time to answer the query with block: ", time.time()-start)
        return df_Block[df_Block["score"]==best].drop(["score"], axis=1)

    elif len(df_SubDistrict)>0:
        # idem
        df_SubDistrict["score"] = df_SubDistrict.apply(lambda x: score_fn(x[cols], 
                                            np.array([state, district, subDistrict, block, GP])), axis=1)
        # Return the closest keys in the state regarding score_fn
        best = df_SubDistrict.sort_values(by=["score"], ascending=False).iloc[0]["score"]
        # print("Time to answer the query with sub district: ", time.time()-start)
        return df_SubDistrict[df_SubDistrict["score"]==best].drop(["score"], axis=1)

    elif len(df_District)>0:
        # idem  
        df_District["score"] = df_District.apply(lambda x: score_fn(x[cols], 
                                            np.array([state, district, subDistrict, block, GP])), axis=1)
        # Return the closest keys in the state regarding score_fn
        best = df_District.sort_values(by=["score"], ascending=False).iloc[0]["score"]
        # print("Time to answer the query with district: ", time.time()-start)
        return df_District[df_District["score"]==best].drop(["score"], axis=1)

    return df_State



    # else:
    #     # We have no matches
    #     df_bis = df.copy()
    #     df_bis["score"] = df_bis.apply(lambda x: score_fn(x[cols], 
    #                                         np.array([state, district, subDistrict, block, GP])), axis=1)
    #     # Return the closest keys in the state regarding score_fn
    #     best = df_bis.sort_values(by=["score"], ascending=False).iloc[0]["score"]
    #     print("Time to answer the query with all data: ", time.time()-start)
    #     return df_bis[df_bis["score"]==best].drop(["score"], axis=1)


def get_cluster(df_query, rule="draw"):

    """
    Return the cluster of a given key using the query dataframe based on a specific rule.
    """
    clusters_values = df_query["Cluster"].value_counts(ascending=False, normalize=True)

    if rule == "max":
        # We return the cluster with the highest number of values
        return np.random.choice(clusters_values.index[clusters_values.values.argmax()], 1)

    elif rule == "draw":
        return np.random.choice(clusters_values.index, 1, p=clusters_values.values)

    else:
        assert False, "Unknown rule"


def fill_submission(df_submission_translated, df_submission, dfs_preds,  states_not_included, rule="max"):
        """
        Fill the submission with the predictions of the model
        """
        Clusters = [-1]*len(df_submission)
        for i in trange(len(df_submission_translated)):
            key = df_submission_translated.iloc[i]["key"]
            state = key.split("_")[0]
            # For these states, we don't have the predictions because 2019 data is not available
            if not state in states_not_included:
                Clusters[i] = get_cluster(get_closest_keys_location(key, dfs_preds), rule=rule)[0]


        df_submission["Cluster"] = Clusters

        return df_submission
