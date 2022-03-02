# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pandas as pd
#import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# from os import isfile, listdir, join
from tqdm import tqdm
import json
# %%
# pathData = "RawData/2017/2017_Andhra Pradesh_Kharif.xlsx"
# df=pd.read_excel(pathData)

# %%
def unify_data(YEAR = 2019, SEASON = 'Kharif') :
    liste_dataframe = []
    monRepertoire = "RawData/"+str(YEAR)
    for f in tqdm(listdir(monRepertoire)) :
        if isfile(join(monRepertoire, f)) and (f[-11:-5] == SEASON or f[-9:-5] == SEASON):
            pathData = "RawData/"+str(YEAR)+"/"+f
            liste_dataframe.append(pd.read_excel(pathData))
    df = pd.concat(liste_dataframe)
    df.to_csv(f'RawDataUnified/RawData_{YEAR}_{SEASON}')

# %%
def add_Loss(df,year=2019):
    """return a new_df with new collumns for Loss and delete the useless collumns after the computation. Only work for the data of 2019"""
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
    for i in range(7) :
        new_df[f'Lp_{year-8+i}'] = vect[i,:]

    #define and add the cumulative monetary loss
    L=np.sum(S*np.maximum(0,threshold-Y),axis=0)/threshold
    new_df["Loss"]=L

    #delete the useless collumns 
    collumns_useless = [f'{y} Yield' for y in np.arange(year-17,year-2+1)]
    collumns_useless.extend(['Sum Insured (Inr)', 'Indemnity Level'])
    new_df = new_df.drop(columns = collumns_useless)
    return new_df


def add_Loss_brouillon(df,year=2019):
    """return a new_df with a new collumn Loss for the data of 2019"""
    Y=np.array([df[f'{y} Yield'] for y in np.arange(year-2-6,year-2+1)])
    theta=np.array(df["Indemnity Level"])
    Y=np.partition(Y,2,axis=0)
    Y=Y[2:,:]
    threshold=np.mean(Y, axis=0)*theta
    S=np.array(df["Sum Insured (Inr)"])

    L=np.sum(S*np.maximum(0,threshold-Y),axis=0)/threshold
    new_df=df
    new_df["Loss"]=L
    return new_df

# %% 
# Pour le clustering des parcelles
def clean_data(df):
    """Clean les data pour l'année 2019"""
    #Suppression de la première colonne inutile (numérotation)
    df = df.drop(columns = ["Unnamed: 0"])
    #Suppression des colonnes sans valeur non nulle
    df = df.drop(columns = ["2018 Yield"])
    df = df.drop(columns = ["2000 Yield"])
    df = df.drop(columns = ["2001 Yield"])
    #Suppression des colonnes inutiles
    df = df.drop(columns = ["Season"])
    df = df.drop(columns = ["Cluster"])
    #consitution de key
    df["Block"] = df["Block"].fillna("")
    df["GP"] = df["GP"].fillna("_")
    df["Sub-District"] = df["Sub-District"].fillna("")
    df["key"] = df["State"]+"_"+df["District"]+"_"+df["Sub-District"]+"_"+df["Block"]+"_"+df["GP"]
    df.key = df.key.astype(str).str.lower()
    #fill na with mean of states
    for state in pd.unique(df["State"]):
        df[df["State"]==state] = df[df["State"]==state].fillna(df[df["State"]==state].mean(numeric_only=True))  
    #Suppression des colonnes des infos administratives et definition de la colonne "key" conformément aux datasets de 03_pred
    df = df.drop(columns = ["State","District","Sub-District","Block","GP"])
    #On remplace les rendements NA restants par leur moyenne
    for year in range(2002,2018):
        col = f"{year} Yield"
        df[col] = df[col].fillna(df[col].mean()) #A FAIRE vérifier que c'est pas abérant de faire ça pour l'anéee 2017. (df["2017 Yield"].isna().sum()) réponse : c'est bcp mais bon...
    for col in ["Area Sown (Ha)","Area Insured (Ha)","SI Per Ha (Inr/Ha)","Sum Insured (Inr)","Indemnity Level"]:
        df[col] = df[col].fillna(df[col].mean())
    df = df.set_index("key")
    return df

# %%
# Pour le clustering des etats
def clean_data_state(df):
    """Nettoie les données pour l'année 2019 en conservant l'appartenance à un Etat"""
    #Suppression de la première colonne inutile (numérotation)
    df = df.drop(columns = ["Unnamed: 0"])
    #Suppression des colonnes sans valeur non nulle
    df = df.drop(columns = ["2018 Yield"])
    df = df.drop(columns = ["2000 Yield"])
    df = df.drop(columns = ["2001 Yield"])
    #Suppression des colonnes inutiles
    df = df.drop(columns = ["Season"])
    df = df.drop(columns = ["Cluster"])
    #consitution de key
    df["Block"] = df["Block"].fillna("")
    df["GP"] = df["GP"].fillna("_")
    df["Sub-District"] = df["Sub-District"].fillna("")
    df["key"] = df["State"]+"_"+df["District"]+"_"+df["Sub-District"]+"_"+df["Block"]+"_"+df["GP"]
    df.key = df.key.astype(str).str.lower()
    #fill na with mean of states
    for state in pd.unique(df["State"]):
        df[df["State"]==state] = df[df["State"]==state].fillna(df[df["State"]==state].mean(numeric_only=True))  
    #Suppression des colonnes des infos administratives et definition de la colonne "key" conformément aux datasets de 03_pred
    df = df.drop(columns = ["District","Sub-District","Block","GP","Crop"])
    

    #On remplace les rendements NA restants par leur moyenne
    for year in range(2002,2018):
        col = f"{year} Yield"
        df[col] = df[col].fillna(df[col].mean()) #A FAIRE vérifier que c'est pas abérant de faire ça pour l'anéee 2017. (df["2017 Yield"].isna().sum()) réponse : c'est bcp mais bon...
    for col in ["Area Sown (Ha)","Area Insured (Ha)","SI Per Ha (Inr/Ha)","Sum Insured (Inr)","Indemnity Level"]:
        df[col] = df[col].fillna(df[col].mean())
    df = df.set_index("key")
    return df

# %%
# BROUILLON 

def clean_data_brouillon(df):
    #Suppresion des colonnes sans valeur non nulle
    df = df.drop(columns = ["2000 Yield","2001 Yield","2002 Yield","2003 Yield","2004 Yield","2005 Yield","2018 Yield"])
    #Suppression des colonnes inutiles
    df = df.drop(columns = ["State","Sub-District","Block","GP"])
    #On remplace les rendements nuls par leur moyenne
    for year in range(2006,2017):
        col = f"{year} Yield"
        df[col] = df[col].fillna(df[col].mean())
    return df
# %%
# new_df=add_Loss(clean_data(df),2015)
# # %%
# new_df.info()
# # %%
# new_df.columns
# # %%


def processing_data(data) :
    dataclean = add_Loss(clean_data(data),year=2019)

#%%
def regroupe_crop(df):
    """Regroupe les crops du datasets df"""
    crop_to_merge = {}
    crops = pd.unique(df["Crop"])
    for crop in crops:
        if crop[:-4] in crops:
            crop_to_merge[crop] = crop[:-4]
        elif crop[:-7] in crops:
            crop_to_merge[crop] = crop[:-7] 
        else:
            crop_to_merge[crop] = crop
    crop_to_merge['Ragi IRR'] = "Ragi Un-IRR"
    crop_to_merge['ONION IRR'] = 'Onion'
    crop_to_merge['Paddy II'] = 'Paddy'
    crop_to_merge['Potato Un-IRR'] = 'Potato IRR'
    crop_to_merge['Chilli IRR'] = 'Chilli Un-IRR'
    
    df['Crop'] = df["Crop"].map(crop_to_merge)
    return df

# %%
def add_climate_clusters(df,rabi, lower = False):
    """ Ajoute les clusters climatiques au dataframe df contenant une colonne State, rabi est un booléen indiquant la saison"""
    if rabi:
        saison = "rabi"
    else:
        saison = "kharif"
    path = "../../Outputs/Predictions/climate_clusters_"+saison+".json"
    with open(path) as json_file:
        dict = json.load(json_file)
        if lower :
            dict =  {k.lower(): v for k, v in dict.items()}
    new_df = df.copy()
    new_df["climate_clusters"] = new_df["State"].map(dict)
    return new_df

def add_crop_categories(df,rabi):
    """ Ajoute les catégories de crop au dataframe df contenant une colonne Crop, rabi est un booléen indiquant la saison"""
    if rabi:
        saison = "Rabi"
    else:
        saison = "Kharif"
    path = "../../Outputs/Predictions/Crop_"+saison+".json"
    with open(path) as json_file:
        dict = json.load(json_file)
    new_df = df.copy()
    new_df["crop_categories"] = new_df["Crop"].map(dict)
    return new_df

# %%
