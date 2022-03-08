
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import pandas as pd
from kmodes.kprototypes import KPrototypes
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

from src.utils import add_Loss, add_climate_clusters, add_crop_categories
from src.clean import clean_data_state, regroup_crop
from src.extractClusters import get_clusters_state


# ----------------------------------------------------------------------------------------------------------------------

date_today = dt.datetime.now().strftime("%Y-%m-%d")

argparser = argparse.ArgumentParser(description="Performs a states clustering")
argparser.add_argument("--name_id", type=str, default=str(date_today), help="Id to be added at the end of the file name")
argparser.add_argument("--output_dir", type=str, default="predictions/state/", help="Output directory")
argparser.add_argument("--nb_clusters", type=int, default=4, help="Number of clusters")
argparser.add_argument("--pen_state", type=float, default=10E10, help="State penalization")
argparser.add_argument("--pen_crop", type=float, default=10, help="Crop penalization")
argparser.add_argument("--pen_climate", type=float, default=10, help="Climate penalization")
args = argparser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------

def import_data(season):
    """
    ## Description
    Import and clean data
    
    ## Returns
    A tuple : (dataset for clustering, dataset for DB criterion)
    """
    #Select the dataset of one season of one year
    year = 2019

    #Path to the dataset
    pathData = f"data/merged_data/RawData_{year}_{season}.csv"
    df= pd.read_csv(pathData)

    #Clean data, add loss, crop categories and climate clusters to data
    pathEmbeddings = Path('output/embeddings/')
    df=add_Loss(clean_data_state(add_crop_categories(regroup_crop(df), season, pathEmbeddings)))
    df = add_climate_clusters(df, season, pathEmbeddings)

    #data for clustering with k-prototypes 
    columns = [f'Lp_{i}' for i in range(2011,2018)] # colonnes : only production losses
    #adding other datas
    columns.append("crop_categories")
    columns.append("climate_clusters")
    columns.append("State")

    #Data for Davis_Bouldin criteria 
    columns_db = [f'Lp_{i}' for i in range(2011,2018)]

    return df[columns], df[columns_db]

def main():
    # retrieve all args
    name_id = args.name_id
    output_dir = args.output_dir
    nb_clusters = args.nb_clusters
    pen_state = args.pen_state
    pen_crop = args.pen_crop
    pen_climate = args.pen_climate

    def categorical_dissimilarity(a, b, **_):
        """
        ## Description
        Dissimilarity function with state penalization for k-prototypes
        """
        return (a[:,0] != b[0])*pen_crop + (a[:,1] != b[1])*pen_climate + (a[:,2] != b[2])*pen_state


    categorical_columns = [7,8,9] #index of the categorical variables

    #Rabi
    print("Importing data...")
    data_R, _ = import_data('Rabi')

    #Clustering with nb_clusters clusters
    print("Launching the clustering for Rabi...")
    kproto = KPrototypes(n_clusters=nb_clusters, init='Cao', n_jobs=-1, cat_dissim=categorical_dissimilarity, n_init=1, random_state=1)
    clusters_R = kproto.fit_predict(data_R, categorical=categorical_columns)
    dict_state_R = get_clusters_state(data_R,clusters_R)

    #Kharif
    data_K, _ = import_data('Kharif')

    #Clustering with nb_clusters clusters
    print("Launching the clustering for Kharif...")
    kproto = KPrototypes(n_clusters=nb_clusters, init='Cao', n_jobs=-1, cat_dissim=categorical_dissimilarity, n_init=1, random_state=1)
    clusters_K = kproto.fit_predict(data_K, categorical=categorical_columns)
    dict_state_K = get_clusters_state(data_K,clusters_K)

    dict_state_K["Assam"] = dict_state_K['Jharkhand']
    dict_state_K["Tamil Nadu"] = dict_state_K['Andhra Pradesh']
    states = list(dict_state_K.keys())
    labels_K = [dict_state_K[state] for state in states]
    dict_state_R["Jharkhand"] = dict_state_R['Chhattisgarh']
    dict_state_R["Assam"] = dict_state_R['Jharkhand']
    dict_state_R["Uttarakhand"]= dict_state_R['Uttar Pradesh']
    labels_R = [dict_state_R[state] for state in states]
    
    pd.DataFrame({'State':states,'Kharif':labels_K,'Rabi':labels_R}).to_csv(Path(output_dir) / f"States_pred{name_id}.csv")
    print("Done")

if __name__ == "__main__":
    main()
    