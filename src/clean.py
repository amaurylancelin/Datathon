import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def unify_data(root_data, root_unified_data, year=2019, season='Kharif') :
    """
    ## Description
    Unifying data of all states from a specific years and season.
    
    ## Parameters
    - root_data (str) : path to the data folder
    - root_unified_data (str) : path to the unified data folder
    """

    list_dataframe = []
    root_year = Path(root_data) / str(year)

    for f in tqdm(os.listdir(root_year)) :
        if os.isfile(os.join(root_year, f)) and (f[-11:-5] == season or f[-9:-5] == season):
            pathData = root_year / f
            list_dataframe.append(pd.read_excel(pathData))

    df = pd.concat(list_dataframe)
    
    os.mkdir(root_unified_data, exists=True)
    
    df.to_csv(Path(root_unified_data) / f'RawData_{year}_{season}')

def regroup_crop(df):
    """
    ## Description
    Grouping crop by crop type.

    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the crops to be regrouped
    """
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

def clean_data(df, year=2019):
    """
    ## Description
    Cleaning data of a specific year.

    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the crops to be cleaned
    """

    #Suppression de la première colonne inutile (numérotation)
    df = df.drop(columns=["Unnamed: 0"])
    #Suppression des colonnes sans valeur non nulle
    df = df.drop(columns=["2018 Yield"])
    df = df.drop(columns=["2000 Yield"])
    df = df.drop(columns=["2001 Yield"])
    #Suppression des colonnes inutiles
    df = df.drop(columns=["Season"])
    df = df.drop(columns=["Cluster"])

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
    df = df.drop(columns=["State","District","Sub-District","Block","GP"])
    #On remplace les rendements NA restants par leur moyenne
    for year in range(2002,year-1):
        col = f"{year} Yield"
        df[col] = df[col].fillna(df[col].mean()) #A FAIRE vérifier que c'est pas abérant de faire ça pour l'anéee 2017. (df["2017 Yield"].isna().sum()) réponse : c'est bcp mais bon...
    for col in ["Area Sown (Ha)","Area Insured (Ha)","SI Per Ha (Inr/Ha)","Sum Insured (Inr)","Indemnity Level"]:
        df[col] = df[col].fillna(df[col].mean())
    df = df.set_index("key")
    return df

# Pour le clustering des etats
def clean_data_state(df):
    """Nettoie les données pour l'année 2019 en conservant l'appartenance à un Etat"""
    #Suppression de la première colonne inutile (numérotation)
    df = df.drop(columns=["Unnamed: 0"])
    #Suppression des colonnes sans valeur non nulle
    df = df.drop(columns=["2018 Yield"])
    df = df.drop(columns=["2000 Yield"])
    df = df.drop(columns=["2001 Yield"])
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
