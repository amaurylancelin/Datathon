import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Define the root of our project
root = "/users/eleves-b/2019/maxime.bonnin/Datathon/"
os.chdir(root)

from Environnement.extractClusters import get_closest_keys_scoring, score_fn, get_cluster

season = "Rabi"
# Def the path of the clusters file
pathPreds = root + f"kmeans_labels_{season}_14-02"

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

df_submission = pd.read_csv(pathSubmission)

def fill_submission(df_submission, df_preds, rule="max"):
    """
    Fill the submission with the predictions of the model
    """
    Clusters = []
    for i in tqdm(range(len(df_submission))):
        key = df_submission.iloc[i]["key"]
        Clusters.append(get_cluster(get_closest_keys_scoring(key, df_preds, score_fn=score_fn), rule=rule))
    return df_submission


df_submission = fill_submission(df_submission, df_preds, rule="draw")

df_submission.to_csv(pathSubFinal)
