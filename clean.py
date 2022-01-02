import collections

import pandas as pd
import numpy as np


from tqdm import trange

INDEX = [
    "Area Sown (Ha)",
    "Area Insured (Ha)",
    "SI Per Ha (Inr/Ha)",
    "Sum Insured (Inr)",
    "Indemnity Level",
]

INDEX.extend([f"{year} Yield" for year in range(2000,2019)])


def main_clean(df, transformation=None):
    """
    This function cleans the dataframe and returns a cleaned dataframe.
    """

    df_new = df.copy()
    stats = compute_mean_by_crop(df_new)
    df_new = precleaning_area_sown(df_new)
    df_new = precleaning_yield(df_new)
    df_new = fill_NaN(df_new, stats)

    if transformation=="normalization":
        # Can be modified to take into account
        # other normalization methods
        df_new = normalization(df_new)

    elif transformation=="standardization":
        df_new = standardization(df_new)

    elif isinstance(transformation, collections.abc.Callable):
        df_new = transformation(df_new)

    return df_new


def compute_mean_by_crop(df):
    """
    This function computes the mean of the different index by crop.
    """

    stats = {}

    df = df.copy()
    df["Crop"] = df["Crop"].str.lower()
    
    crops = df["Crop"].unique()

    for crop in crops:
        # Be careful modifications on it are not applied to the original df
        sub_df = df.query(f"Crop == '{crop}'") 
        stats[crop] = {}
        for index in INDEX:
            stats[crop][index] = {}
            try:
                N = int(len(sub_df) - pd.isna(sub_df[index]).sum())
            
            except KeyError:
                N = 0 

            if N == 0:
                stats[crop][index]["average"] = np.nan
                stats[crop][index]["N"] = 0

            else:
                stats[crop][index]["average"] = sub_df[index].mean()
                stats[crop][index]["N"] = N

    results = {}

    # Adding mean by crop for every index
    for crop in stats.keys():
        results[crop] = {}

        for index in stats[crop].keys():
            if index != "N":
                if stats[crop][index]["N"] == 0:
                    results[crop][index] = np.nan
                else:
                    results[crop][index] = stats[crop][index]["average"]

    #Adding overall mean for every index
    results["overall"] = {}
    for crop in stats.keys():
        for index in stats[crop].keys():
            if index not in results["overall"]:
                results["overall"][index] = {}
                results["overall"][index]["N"] = 0
                results["overall"][index]["sum"] = 0  

            results["overall"][index]["N"] += stats[crop][index]["N"]
            results["overall"][index]["sum"] += stats[crop][index]["average"]*stats[crop][index]["N"]

    for index in results["overall"].keys():
        if results["overall"][index]["N"] == 0:
            results["overall"][index] = np.nan
        else:
            results["overall"][index] = results["overall"][index]["sum"] / results["overall"][index]["N"]
            
    return results

def filler(df, stats, col):
    """
    This function adds the mean of the different index by crop.
    """
    df_copy = df.copy()
    df_copy["Bool"] = df_copy[col].isna().astype(int)

    for crop in df["Crop"].unique():
        
        index = df_copy.query(f"Crop == '{crop}' and Bool == 1").index
        
        df_copy.loc[index,col] = stats[crop][col]
        
    df[col] = df_copy[col]

    return df



def precleaning_area_sown(df):
    """
    This function cleans the column Area Sown (Ha) and put NaN if the value is not numeric.
    It may contains non numeric values.
    """
    newValues = []
    for value in df["Area Sown (Ha)"]:
        try:
            value = float(value)
            newValues.append(value)
        except ValueError:
            newValues.append(np.NaN)
    df["Area Sown (Ha)"] = newValues
    return df
    

def precleaning_yield(df):
    
    """
    This function cleans all columns '20XX Yield' and put NaN if the value is not numeric.
    It may contains non numeric values.
    """
    for year in range(2000,2019):
        
        try:
            df[f"{year} Yield"]
        except KeyError:
            continue
        
        newValues = []
        for value in df[f"{year} Yield"]:
            try:
                value = float(value)
                newValues.append(value)
            except ValueError:
                newValues.append(np.NaN)
        df[f"{year} Yield"] = newValues

        
        
    return df

def fill_NaN(df, stats):

    """
    This function fills the missing vlaues for every index 
    using statistics on crop and overall statistics if needed.
    """
    df["Crop"] = df["Crop"].str.lower()

    # filling index ...
    for index in INDEX:
        try:
            df = filler(df, stats, index)

        except KeyError:
            df[index] = [np.nan] * len(df)
            continue


    for index in stats["overall"].keys():
        if index != "N":
            try:
                df[index].fillna(stats["overall"][index], inplace=True)

            except KeyError:
                continue

    return df


def normalization(df):
    """
    This function normalizes all index.
    """
    for index in INDEX:
        try:
            df[index] = (df[index] - df[index].min()) / (df[index].max() - df[index].min())
        except KeyError:
            continue

    return df

def standardization(df):
    """
    This function standardizes all index.
    """
    for index in INDEX:
        try:
            df[index] = (df[index] - df[index].mean()) / df[index].std()
        except KeyError:
            continue

    return df