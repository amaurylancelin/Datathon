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

    isinstance()
    return df_new



def compute_mean_by_crop(df):
    """
    This function computes the mean of the different index by crop.
    """

    stats = {}

    df = df.copy()
    df["Crop"] = df["Crop"].str.lower()
    
    for i in trange(df.shape[0]):
        crop = df.iloc[i]["Crop"]
        if crop not in stats:
            stats[crop] = {}
            for index in INDEX:
                try:
                    stats[crop][index] = {"N": 0, "sum": 0} 

                except KeyError:
                    continue

            # stats[crop]["Area Sown (Ha)"] = {"N": 0, "sum": 0} 
            # stats[crop]["Area Insured (Ha)"] = {"N": 0, "sum": 0} 
            # stats[crop]["SI Per Ha (Inr/Ha)"] = {"N": 0, "sum": 0} 
            # stats[crop]["Sum Insured (Inr)"] = {"N": 0, "sum": 0} 
            # stats[crop]["Indemnity Level"] = {"N": 0, "sum": 0} 
            

            # for year in range(2000,2019):
            #     try:
            #         stats[crop][f"{year} Yield"] = {"N": 0, "sum": 0} 
            #     except KeyError:
            #         continue

        for index in INDEX:
            try:
                value = df.iloc[i][index]

            except KeyError:
                continue

            if not pd.isna(value):
                stats[crop][index]["N"] += 1
                stats[crop][index]["sum"] += value
        # area_sown = df.iloc[i]["Area Sown (Ha)"]
        # area_insured = df.iloc[i]["Area Insured (Ha)"]
        # si_per_ha = df.iloc[i]["SI Per Ha (Inr/Ha)"]
        # sum_insured = df.iloc[i]["Sum Insured (Inr)"]
        # indemnity_level = df.iloc[i]["Indemnity Level"]

        # if not pd.isna(area_sown):
        #     stats[crop]["Area Sown (Ha)"]["sum"] += area_sown
        #     stats[crop]["Area Sown (Ha)"]["N"] += 1
        
        # if not pd.isna(area_insured):
        #     stats[crop]["Area Insured (Ha)"]["sum"] += area_insured
        #     stats[crop]["Area Insured (Ha)"]["N"] += 1

        # if not pd.isna(si_per_ha):
        #     stats[crop]["SI Per Ha (Inr/Ha)"]["sum"] += si_per_ha
        #     stats[crop]["SI Per Ha (Inr/Ha)"]["N"] += 1

        # if not pd.isna(sum_insured):
        #     stats[crop]["Sum Insured (Inr)"]["sum"] += sum_insured
        #     stats[crop]["Sum Insured (Inr)"]["N"] += 1

        # if not pd.isna(indemnity_level):
        #     stats[crop]["Indemnity Level"]["sum"] += indemnity_level
        #     stats[crop]["Indemnity Level"]["N"] += 1
        

        # for year in range(2000,2019):
        #     try:
        #         if not pd.isna(df.iloc[i][f"{year} Yield"]):
        #             stats[crop][f"{year} Yield"]["sum"] += df.iloc[i][f"{year} Yield"]
        #             stats[crop][f"{year} Yield"]["N"] += 1
        #     except KeyError:
        #         continue

    results = {}

    # Adding mean by crop for every index
    for crop in stats.keys():
        results[crop] = {}

        for index in stats[crop].keys():
            if index != "N":
                if stats[crop][index]["N"] == 0:
                    results[crop][index] = np.nan
                else:
                    results[crop][index] = stats[crop][index]["sum"] / stats[crop][index]["N"]

    #Adding overall mean for every index
    results["overall"] = {}
    for crop in stats.keys():
        for index in stats[crop].keys():
            if index not in results["overall"]:
                results["overall"][index] = {}
                results["overall"][index]["N"] = 0
                results["overall"][index]["sum"] = 0  

            results["overall"][index]["N"] += stats[crop][index]["N"]
            results["overall"][index]["sum"] += stats[crop][index]["sum"]

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
    for i in range(df.shape[0]):
        crop = df.loc[i, "Crop"].lower()
        if pd.isna(df.loc[i, col]) and not pd.isna(stats[crop][col]):
            df.loc[i, col] = stats[crop][col]
    

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

    # df = filler(df, stats, "Area Sown (Ha)")
    # df = filler(df, stats, "Area Insured (Ha)")
    # df = filler(df, stats, "SI Per Ha (Inr/Ha)")
    # df = filler(df, stats, "Sum Insured (Inr)")
    # df = filler(df, stats, "Indemnity Level")

    # # filling yields 
    # for year in range(2000,2019):
    #     try:
    #         df = filler(df, stats, f"{year} Yield")
    #     except KeyError:
    #         df[f"{year} Yield"] = [np.nan]*len(df)
    #         continue

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