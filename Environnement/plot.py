import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#admLvl stands for administrative level : states, districts,...

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