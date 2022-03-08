import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns

try:
    import geopandas as gpd
    import_error = False
except ImportError:
    import_error = True
    pass

from src.clean import (
    clean_data,
    regroup_crop
    )
from src.utils import (
    add_climate_clusters,  
    add_crop_categories,
    add_Loss,
)

# from pathlib import Path
# import sys
# os.chdir(Path(sys.path[0]).parent)

# admin_level stands for administrative level : states, districts,...

def get_list_admin_level_cluster(list_admin_level, df_admin_level_cluster, admin_level): 
    
    '''
    ## Description
    Returns a list containing the main cluster for each State or District (according to admin_level).

    ## Parameters
    - list_admin_level (list) : list of the states or districts
    - df_admin_level_cluster (pandas.DataFrame) : dataframe containing the clusters for each state or district
    - admin_level (str) : 'State' or 'District'
    '''
    list_admin_level_cluster = []
    for i in range(len(list_admin_level)):
        l = []
        l.append(list_admin_level[i])
        # print(df_admin_level_cluster[df_admin_level_cluster[admin_level] == list_admin_level[i]]["cluster"].to_numpy())
        if len(df_admin_level_cluster[df_admin_level_cluster[admin_level] == list_admin_level[i]]["cluster"].to_numpy().astype(int)) > 0 :
            l.append(np.bincount(df_admin_level_cluster[df_admin_level_cluster[admin_level] == list_admin_level[i]]["cluster"].to_numpy().astype(int)).argmax())
        list_admin_level_cluster.append(l)
    return list_admin_level_cluster

def plot_on_map(method_labels,pathData,admin_level, cmap = "RdYlGn"):
    
    '''
    ## Description
    Plots the clusters of the parcels on a map of India. For each admin_level, we plot the main cluster.

    ## Parameters
    - method_labels (numpy.array) : labels of the clusters for each parcel
    - pathData (str) : path to the rawdata, useful for the plot
    - admin_level (str) : 'State' or 'District'
    - cmap (str) : color map of the plot
    '''

    if import_error:
        raise ImportError("geopandas is not installed")
    
    labels_df = pd.DataFrame(method_labels, columns=['labels'])
    df_init = pd.read_csv(pathData)
    df_init['cluster'] = labels_df['labels'] 
    df_admin_level_cluster = df_init[[admin_level, 'cluster']]

    list_admin_level = pd.unique(df_admin_level_cluster[admin_level])

    # get the main cluster on each admin_level
    list_admin_level_cluster = get_list_admin_level_cluster(list_admin_level, df_admin_level_cluster, admin_level)
    df_reduced = pd.DataFrame(list_admin_level_cluster, columns=[admin_level, 'Clusters'])

    # get the proper map according to the admin_level
    if admin_level == 'State' :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_1'
    elif admin_level == 'District' :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'

    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admin_level))

    # display the plot
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title('Clustering with k-means, averaged on each '+ admin_level,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Clusters', cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.2', categorical=True, legend=True)


def get_list_admin_level_crop(list_admin_level, list_crops, df_admin_level_crop, admin_level):
    
    '''
    ## Description
    Returns a list containing the main crop category for each State or District (according to admin_level).

    ## Parameters
    - list_admin_level (list) : list of the states or districts
    - list_crops (list) : list of all the crop categories
    - df_admin_level_cluster (pandas.DataFrame) : dataframe containing the clusters for each state or district
    - admin_level (str) : 'State' or 'District'
    '''

    list_admin_level_crop = []
    for i in range(len(list_admin_level)):
        l = []
        l.append(list_admin_level[i])
        if len(df_admin_level_crop[df_admin_level_crop[admin_level] == list_admin_level[i]]["numCrop"].to_numpy().astype(int)) > 0 :
            max_crop_num = np.bincount(df_admin_level_crop[df_admin_level_crop[admin_level] == list_admin_level[i]]["numCrop"].to_numpy().astype(int)).argmax()
            max_crop = df_admin_level_crop.loc[df_admin_level_crop['numCrop'] == max_crop_num, "crop_categories"].iloc[0]
            l.append(max_crop)
        list_admin_level_crop.append(l)
    return list_admin_level_crop

def plot_crops(pathData,admin_level, rabi): 
    
    '''
    ## Description
    Plots the crops of the parcels on a map of India. For each admin_level, we plot the main crop.

    ## Parameters
    - pathData (str) : path to the rawdata, useful for the plot
    - admin_level (str) : 'State' or 'District'
    - rabi (bool) : True if the season is Rabi, False if the season is Kharif
    '''

    if import_error:
        raise ImportError("geopandas is not installed")

    df_init = pd.read_csv(pathData)

    # regroup the crops in categories
    df_admin_level_crop = add_crop_categories(df_init, rabi)

    df_admin_level_crop['numCrop'] = pd.factorize(df_admin_level_crop["crop_categories"])[0]
    print(pd.unique(df_admin_level_crop[df_admin_level_crop['numCrop'] == -1]['Crop']))
    list_admin_level = pd.unique(df_admin_level_crop[admin_level])
    list_crops = pd.unique(df_admin_level_crop["crop_categories"])

    # get the main crop in each admin_level
    list_admin_level_crop = get_list_admin_level_crop(list_admin_level, list_crops, df_admin_level_crop, admin_level)
    df_reduced = pd.DataFrame(list_admin_level_crop, columns=[admin_level, "crop_categories"])

    # get the proper map according to the admin_level
    if admin_level == 'State' :
        map_path = "data/external_data/maps/gadm36_IND_shp/gadm36_IND_1.shp"
        name = 'NAME_1'
    elif admin_level == 'District' :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'

    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admin_level))

    # display the plot
    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.axis('off')
    ax.set_title('Main crop in each '+ admin_level,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='crop_categories', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2', categorical=True, legend=True)


def get_list_admin_level_yield(list_admin_level, df_admin_level_yield, admin_level, bins):
    
    '''
    ## Description
    Returns a list containing the yield for each State or District (according to admin_level).

    ## Parameters
    - list_admin_level (list) : list of the states or districts
    - df_admin_level_cluster (pandas.DataFrame) : dataframe containing the clusters for each state or district
    - admin_level (str) : 'State' or 'District'
    - bins (int) : number of bins
    '''

    list_admin_level_yield = []
    yields = []
    for i in range(len(list_admin_level)):
        l = []
        l.append(list_admin_level[i])
        if len(df_admin_level_yield[df_admin_level_yield[admin_level] == list_admin_level[i]]["2017 Yield"].to_numpy().astype(int)) > 0 :
            mean_yield = np.mean(df_admin_level_yield[df_admin_level_yield[admin_level] == list_admin_level[i]]["2017 Yield"].to_numpy().astype(float))
            yields.append(mean_yield)
            l.append(mean_yield)
        list_admin_level_yield.append(l)
    yields_arr = np.array(yields)
    m = np.min(yields_arr)
    M = np.max(yields_arr)
    step = (M-m)/bins
    for i in range(len(list_admin_level)):
        val = list_admin_level_yield[i][1]
        value = m
        for j in range(bins):
            if val >= (m + j*step) and val < (m + (j+1)*step) :
                value = m + (j+1/2)*step
        list_admin_level_yield[i][1] = value
    return list_admin_level_yield

def plot_yields(pathData,admin_level,bins):
    
    '''
    ## Description
    Plots the yield of the parcels on a map of India. For each admin_level, we plot the bin of the average yield.

    ## Parameters
    - pathData (str) : path to the rawdata, useful for the plot
    - admin_level (str) : 'State' or 'District'
    - bins (int) : number of bins
    '''

    if import_error:
        raise ImportError("geopandas is not installed")

    df_init = pd.read_csv(pathData)
    df_clean = add_Loss(clean_data(df_init))
    df_admin_level_yield = df_clean[[admin_level,'2017 Yield']]

    list_admin_level = pd.unique(df_admin_level_yield[admin_level])

    # get the yield for each admin_level
    list_admin_level_yield = get_list_admin_level_yield(list_admin_level, df_admin_level_yield, admin_level, bins)
    df_reduced = pd.DataFrame(list_admin_level_yield, columns=[admin_level, 'Yield'])

    # get the proper map according to the admin_level
    if admin_level == 'State' :
        map_path = "data/external_data/maps/gadm36_IND_shp/gadm36_IND_1.shp"
        name = 'NAME_1'
    elif admin_level == 'District' :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm2.shp"
        name = 'NAME_2'
    else :
        map_path = "data/external_data/maps/ind_adm_shp/IND_adm3.shp"
        name = 'NAME_3'
 
    map_gdf = gpd.read_file(map_path)
    merged = map_gdf.set_index(name).join(df_reduced.set_index(admin_level))

    # display the plot
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title('Average yield in each '+ admin_level,
                fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column='Yield', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
    

def display_parallel_coordinates_centroids(df, num_clusters):
    
    '''
    ## Description
    Display a parallel coordinates plot for the centroids in df.

    ## Parameters
    - df (pandas.DataFrame) : dataframe containing the clusters
    - num_clusters (int) : number of clusters
    '''
    
    palette = sns.color_palette("bright", 10)

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)   