import json
import numpy as np
import pandas as pd

def add_Loss(df, year=2019):
    
    """
    ## Description
    Return a new dataframe with new columns for the losses
    and delete the useless columns after the computation. 
    
    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the data   

    ## Returns
    - df (pandas.DataFrame) : dataframe with losses as columns (e.g for 2019 Lp_2011, ..., Lp_2017)
    - year (int) : year for which the losses are computed
    """
    
    #define several quantities used in the next formulas
    Yi=np.array([df[f'{y} Yield'] for y in np.arange(year-2-6,year-2+1)])
    theta=np.array(df["Indemnity Level"])
    Y=np.partition(Yi,2,axis=0)
    Y=Y[2:,:]
    threshold=np.mean(Y, axis=0)*theta
    S=np.array(df["Sum Insured (Inr)"])

    #define and add the vector of production loss (used for the db criteria in particular)
    vect=np.maximum(0,threshold-Yi)/threshold
    new_df=df
    for i in range(7):
        new_df[f'Lp_{year-8+i}'] = vect[i,:]

    #define and add the cumulative monetary loss
    L=np.sum(S*np.maximum(0,threshold-Y),axis=0)/threshold
    new_df["Loss"]=L

    #delete the useless columns 
    columns_useless = [f'{y} Yield' for y in np.arange(year-17,year-2+1)]
    columns_useless.extend(['Sum Insured (Inr)', 'Indemnity Level'])
    new_df = new_df.drop(columns = columns_useless)
    return new_df

def add_climate_clusters(df, season):
    """ 
    ## Description
    Adding the climate clusters to the input dataframe.

    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the data
    - season (str) : season to be considered 'Kharif' or 'Rabi'
    - root_clusters (str) : path to the folder containing the clusters    
    """
    
    if season.lower() not in ["rabi", "kharif"]:
        raise ValueError("Season should be 'rabi' or 'kharif'")

    path = f"output/embeddings/climate_clusters_{season.lower()}.json"

    with open(path) as json_file:
        dict = {k.lower(): v for k, v in json.load(json_file).items()}

    new_df = df.copy()
    new_df["climate_clusters"] = new_df["State"].map(dict)
    return new_df

def add_crop_categories(df, season):
    
    """ 
    ## Description
    Adding the crop categories to the input dataframe.

    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the data
    - season (str) : season to be considered 'Kharif' or 'Rabi'
    - root_crops (str) : path to the folder containing the clusters
    """
    if season.lower() not in ["rabi", "kharif"]:
        raise ValueError("Season should be 'rabi' or 'kharif'")

    path = f"output/embeddings/Crop_{season.lower()}.json"
    with open(path) as json_file:
        dict = json.load(json_file)
    
    new_df = df.copy()
    new_df["crop_categories"] = new_df["Crop"].map(dict)
    return new_df
