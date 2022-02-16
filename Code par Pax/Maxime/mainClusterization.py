import pandas as pd
import numpy as np
import os
import sys
from tqdm import trange, tqdm

STATES_NOT_INCLUDED = {"Rabi": ['assam', 'uttarakhand', 'jharkhand'], "Kharif":['assam', 'tamil nadu']}

# Define the root of our project
# root = "/Users/maximebonnin/Documents/Projects/SCOR/Datathon/"

root = "/users/eleves-b/2019/maxime.bonnin/Datathon/"

# adding roots to the system path
sys.path.insert(0, root)

season = "Kharif"
STATES_NOT_INCLUDED = STATES_NOT_INCLUDED[season]

from Environnement.extractClusters import get_closest_keys_scoring, score_fn, get_cluster, get_closest_keys_location

season = "Rabi"
# Def the path of the clusters file
pathPreds = root + f"Outputs/Predictions/kmeans_labels_{season}_14-02"

# Define the predictions needed
pathSubmission = root + f"Data/03_Prediction/GP_Pred_{season}.csv"

pathSubFinal = root + f"Data/03_Prediction/GP_Pred_{season}_final.csv"

df_preds = pd.read_csv(pathPreds)
df_preds["State"] = df_preds["key"].apply(lambda x: x.split("_")[0])
df_preds["District"] = df_preds["key"].apply(lambda x: x.split("_")[1])
df_preds["SubDistrict"] = df_preds["key"].apply(lambda x: x.split("_")[2])
df_preds["Block"] = df_preds["key"].apply(lambda x: x.split("_")[3])
df_preds["GP"] = df_preds["key"].apply(lambda x: x.split("_")[4])
df_preds["Cluster"] = df_preds["0"]

dfs_preds = {state:df_preds.loc[df_preds["State"]== state] for state in df_preds["State"].unique()}

df_submission = pd.read_csv(pathSubmission)

def fill_submission(df_submission, dfs_preds, rule="max"):
    """
    Fill the submission with the predictions of the model
    """
    Clusters = [-1]*len(df_submission)
    # print(len(Clusters), len(df_submission))
    for i in trange(len(df_submission)):
        key = df_submission.iloc[i]["key"]
        state = key.split("_")[0]
        if not state in STATES_NOT_INCLUDED:
            # Clusters[i] = get_cluster(get_closest_keys_scoring(key, dfs_preds, score_fn=score_fn), rule=rule)[0]
            Clusters[i] = get_cluster(get_closest_keys_location(key, dfs_preds), rule=rule)[0]

        
        # if i==3:
        #     break

    df_submission["Cluster"] = Clusters

    return df_submission


df_submission = fill_submission(df_submission, dfs_preds, rule="draw")

df_submission.to_csv(pathSubFinal)
