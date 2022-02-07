# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from os import isfile, listdir, join
from tqdm import tqdm

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
        new_df[f'Lp_{2011+i}'] = vect[i,:]

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
    df = df.drop(columns = ["District","Sub-District","Block","GP"])
    
    le = LabelEncoder()
    df["State"] = le.fit_transform(df["State"])
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


def get_liste_admLvl_cluster(list_admLvl, df_admLvl_cluster, admLvl):
    liste_admLvl_cluster = []
    for i in range(len(list_admLvl)):
        l = []
        l.append(list_admLvl[i])
        #print(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy())
        if len(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy().astype(int)) > 0 :
            l.append(np.bincount(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy().astype(int)).argmax())
        liste_admLvl_cluster.append(l)
    return liste_admLvl_cluster

# plot clusters on map of India
# typiquement admLvl = 'District', method_labels = kmeans.labels_ par exemple
# pathData renvoie vers les données brutes initiales
def plot_on_map(method_labels,pathData,admLvl):      
    labels_df = pd.DataFrame(method_labels, columns=['labels'])
    df_init = pd.read_csv(pathData)
    df_init['cluster'] = labels_df['labels'] 
    df_admLvl_cluster = df_init[[admLvl, 'cluster']]

    list_admLvl = pd.unique(df_admLvl_cluster[admLvl])

    list_admLvl_cluster = get_liste_admLvl_cluster(list_admLvl, df_admLvl_cluster, admLvl)
    df_reduced = pd.DataFrame(list_admLvl_cluster, columns=[admLvl, 'Clusters'])

    if admLvl == 'State' :
        map_path = "maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_1'
    elif admLvl == 'District' :
        map_path = "maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'

    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admLvl))

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title('Clustering with k-means, averaged on each '+ admLvl,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Clusters', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
# %%


def get_liste_admLvl_crop(list_admLvl, df_admLvl_crop, admLvl):
    liste_admLvl_crop = []
    for i in range(len(list_admLvl)):
        l = []
        l.append(list_admLvl[i])
        #print(df_admLvl_crop[df_admLvl_crop[admLvl] == list_admLvl[i]]["Crop"].to_numpy())
        if len(df_admLvl_crop[df_admLvl_crop[admLvl] == list_admLvl[i]]["Crop"].to_numpy().astype(int)) > 0 :
            l.append(np.bincount(df_admLvl_crop[df_admLvl_crop[admLvl] == list_admLvl[i]]["Crop"].to_numpy().astype(int)).argmax())
        liste_admLvl_crop.append(l)
    return liste_admLvl_crop

def plot_crops(pathData,admLvl):      
    df_init = pd.read_csv(pathData)
    df_admLvl_crop = df_init[[admLvl, 'Crop']]
    df_admLvl_crop['Crop'] = pd.factorize(df_admLvl_crop['Crop'])[0]
    # print(df_admLvl_crop)
    list_admLvl = pd.unique(df_admLvl_crop[admLvl])

    list_admLvl_crop = get_liste_admLvl_crop(list_admLvl, df_admLvl_crop, admLvl)
    df_reduced = pd.DataFrame(list_admLvl_crop, columns=[admLvl, 'Crop'])

    if admLvl == 'State' :
        map_path = "maps/gadm36_IND_shp/gadm36_IND_1.shp"
        name = 'NAME_1'
    elif admLvl == 'District' :
        map_path = "maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'

    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admLvl))

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title('Main crop in each '+ admLvl,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Crop', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)

#%%

# il faut modifier un peu clean data pour que ce plot fonctionne
def get_liste_admLvl_yield(list_admLvl, df_admLvl_yield, admLvl, K):
    liste_admLvl_yield = []
    yields = []
    for i in range(len(list_admLvl)):
        l = []
        l.append(list_admLvl[i])
        #print(df_admLvl_yield[df_admLvl_yield[admLvl] == list_admLvl[i]]["2017 Yield"].to_numpy())
        if len(df_admLvl_yield[df_admLvl_yield[admLvl] == list_admLvl[i]]["2017 Yield"].to_numpy().astype(int)) > 0 :
            mean_yield = np.mean(df_admLvl_yield[df_admLvl_yield[admLvl] == list_admLvl[i]]["2017 Yield"].to_numpy().astype(float))
            yields.append(mean_yield)
            l.append(mean_yield)
        liste_admLvl_yield.append(l)
    yields_arr = np.array(yields)
    m = np.min(yields_arr)
    M = np.max(yields_arr)
    step = (M-m)/K
    for i in range(len(list_admLvl)):
        val = liste_admLvl_yield[i][1]
        value = m
        for j in range(K):
            if val >= (m + j*step) and val < (m + (j+1)*step) :
                value = m + (j+1/2)*step
        liste_admLvl_yield[i][1] = value
    return liste_admLvl_yield

def plot_yields(pathData,admLvl,K):      
    df_init = pd.read_csv(pathData)
    df_clean = add_Loss(clean_data(df_init))
    df_admLvl_yield = df_clean[[admLvl,'2017 Yield']]

    #m = df_admLvl_yield['2017 Yield'].min()
    #M = df_admLvl_yield['2017 Yield'].max()

    # print(df_admLvl_yield)
    list_admLvl = pd.unique(df_admLvl_yield[admLvl])

    list_admLvl_yield = get_liste_admLvl_yield(list_admLvl, df_admLvl_yield, admLvl, K)
    df_reduced = pd.DataFrame(list_admLvl_yield, columns=[admLvl, 'Yield'])
    print(df_reduced)

    if admLvl == 'State' :
        map_path = "maps/gadm36_IND_shp/gadm36_IND_1.shp"
        name = 'NAME_1'
    elif admLvl == 'District' :
        map_path = "maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'

    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admLvl))

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title('Average yield in each '+ admLvl,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Yield', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)



    def processing_data(data) :
        dataclean = add_Loss(clean_data(data),year=2017)
