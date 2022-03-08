import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import datetime as dt

from src.extractClusters import fill_submission


date_today = dt.datetime.now().strftime("%Y-%m-%d")

STATES_NOT_INCLUDED = {"Rabi": ['assam', 'uttarakhand', 'jharkhand'], "Kharif":['assam', 'tamil nadu']}


argparser = argparse.ArgumentParser(description="Fill the submission with the predictions of the model by finding the closest cluster")
argparser.add_argument("--season", type=str, help="Season to be filled", required=True)
argparser.add_argument("--preds_path", type=str, help="Path of the predictions", required=True)
argparser.add_argument("--name_id", type=str, default=str(date_today), help="Id to be added at the end of the file name")
argparser.add_argument("--output_dir", type=str, default="Outputs/Results/", help="Output directory")
argparser.add_argument("--empty_file_dir", type=str, default="Data/03_Prediction/", help="Empty file to fill")
args = argparser.parse_args()


def main():
    # retrieve all args
    season = args.season
    name_id = args.name_id
    pathPreds = args.preds_path
    output_dir = args.output_dir
    empty_file_dir = args.empty_file_dir
    
    print("Season to be filled...",season)

    states_not_included_season = STATES_NOT_INCLUDED[season]

    # Define the predictions needed
    pathEmptySubmission = Path(empty_file_dir) /f"GP_Pred_{season}_ID.csv"
    # The path to the same file as above but translated into english to increase the number of perfect matches
    pathEmptySubmissionTranslated = Path(empty_file_dir) / f"GP_Pred_{season}_ID_translated.csv"
    # The path to the file to be filled with the predictions of the model
    pathOutputSubmission = Path(output_dir) / f"GP_Pred_{season}_{name_id}.csv"

    # The file containing the predictions of the model translated
    df_preds = pd.read_csv(pathPreds)
    # The submitted translated file of the model translated
    df_EmptySubmissionTranslated = pd.read_csv(pathEmptySubmissionTranslated)
    # The submitted file of the model
    df_EmptySubmission = pd.read_csv(pathEmptySubmission)

    df_preds["State"] = df_preds["key"].apply(lambda x: x.split("_")[0])
    df_preds["District"] = df_preds["key"].apply(lambda x: x.split("_")[1])
    df_preds["SubDistrict"] = df_preds["key"].apply(lambda x: x.split("_")[2])
    df_preds["Block"] = df_preds["key"].apply(lambda x: x.split("_")[3])
    df_preds["GP"] = df_preds["key"].apply(lambda x: x.split("_")[4])
    df_preds["Cluster"] = df_preds["0"]

    dfs_preds = {state:df_preds.loc[df_preds["State"]== state] for state in df_preds["State"].unique()}
    
    df_submission = fill_submission(df_EmptySubmissionTranslated, df_EmptySubmission, dfs_preds, states_not_included_season, rule="draw")

    df_submission.to_csv(pathOutputSubmission)


if __name__ == "__main__":
    main()