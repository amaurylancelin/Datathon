# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
# import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


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
def add_Loss(df,year=2017):
    """return a new_df with a new collumn Loss for the data of 2019"""
    Y=np.array([df[f'{y} Yield'] for y in np.arange(year-6,year+1)])
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

def get_liste_district_cluster(list_admLvl, df_admLvl_cluster, admLvl):
    liste_admLvl_cluster = []
    for i in range(len(list_admLvl)):
        l = []
        l.append(list_admLvl[i])
        print(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy())
        if len(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy().astype(int)) > 0 :
            l.append(np.bincount(df_admLvl_cluster[df_admLvl_cluster[admLvl] == list_admLvl[i]]["cluster"].to_numpy().astype(int)).argmax())
        liste_admLvl_cluster.append(l)
    return liste_admLvl_cluster


#plot clusters on map of India

def plot_on_map(method_labels,pathData,admLvl):      # typiquement admLvl = 'District'
    labels_df = pd.DataFrame(method_labels, columns=['labels'])
    df_init = pd.read_csv(pathData)
    df_init['cluster'] = labels_df['labels'] 
    df_admLvl_cluster = df_init[['District', 'cluster']]

    list_admLvl = pd.unique(df_admLvl_cluster[admLvl])

    list_district_cluster = get_liste_district_cluster(list_admLvl, df_admLvl_cluster, admLvl)
    df_reduced = pd.DataFrame(list_district_cluster, columns=[admLvl, 'Clusters'])

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
    ax.set_title('Clustering with k-means, averaged on each district',
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Clusters', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)