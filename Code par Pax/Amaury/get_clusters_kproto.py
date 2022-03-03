import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm, trange
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from prince import FAMD, PCA
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
# import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


#Our own librabries
import sys
sys.path.insert(1, '../../Environnement/')
from utils import clean_data, add_Loss, regroupe_crop
from utils import add_crop_categories, add_climate_clusters
from plot import plot_on_map, display_parallel_coordinates, display_parallel_coordinates_centroids
import clean
import merge



#SELECT THE DATE
date ='02-03'

#Select the dataset of one season of one year
YEAR = 2019
SEASON = "Rabi" # or "Kharif" 

#Path to the dataset
pathData_R = f"../../Data/RawDataUnified/RawData_{YEAR}_Rabi"
pathData_K= f"../../Data/RawDataUnified/RawData_{YEAR}_Kharif"


df_R = pd.read_csv(pathData_R)
df_K = pd.read_csv(pathData_K)


#Clean an add loss to data
df_R=add_Loss(clean_data(df_R))
df_K=add_Loss(clean_data(df_K))


#NORMALISATION OF DATA (uses : for FAMD and for computing the DB criteria)
data_R=df_R.copy(deep=True)
data_K=df_K.copy(deep=True)
scale = StandardScaler()
data_R.loc[:,data_R.columns !='Crop']=scale.fit_transform(data_R.loc[:,data_R.columns !='Crop'])
data_K.loc[:,data_K.columns !='Crop']=scale.fit_transform(data_K.loc[:,data_K.columns !='Crop'])

#data for Davis_Bouldin criteria 
collumns_db = [f'Lp_{i}' for i in range(2011,2018)]
data_R_db=data_R[collumns_db]
data_K_db=data_K[collumns_db]



def find_state(key) : 
    return key.split('_')[0]

collumns_db= [f'Lp_{i}' for i in range(2011,2018)] 
collumns = collumns_db + ['Crop']

Rabi = regroupe_crop(data_R)
Rabi = Rabi[collumns]
Rabi = Rabi.reset_index()
Rabi['State']=(Rabi.reset_index())['key'].map(find_state)
Rabi= Rabi.set_index('key')
Rabi = add_climate_clusters(Rabi, rabi = True, lower=True)
Rabi = add_crop_categories(Rabi, rabi = True).drop(columns=['Crop', 'State'])

Kharif = regroupe_crop(data_K)
Kharif = Kharif[collumns]
Kharif = Kharif.reset_index()
Kharif['State']=(Kharif.reset_index())['key'].map(find_state)
Kharif= Kharif.set_index('key')
Kharif = add_climate_clusters(Kharif, rabi = False, lower=True)
Kharif = add_crop_categories(Kharif, rabi = False).drop(columns=['Crop', 'State'])


def cat_diss_pen(pen) :
    def categorical_dissimilarity(a, b, **_):
        """Dissimilarity function with penalization"""
        return (a[:,0] != b[0])*pen[0] + (a[:,1] != b[1])*pen[1]
    return categorical_dissimilarity


def compute_clusters(Season_df, nb_clusters=7, pen =[1,1], nb_samples=200, all_data=False):
    if all_data == True :
        Season_test = Season_df
    else :
        Season_test=Season_df.sample(frac=1, random_state=0)[:nb_samples]
    
    #'7' -> 'climate_clusters' and '8' -> 'crop_categories'
    categorical_columns = [7,8]
    categorical_dissimilarity = cat_diss_pen(pen)
    kproto = KPrototypes(n_clusters= nb_clusters, init='Cao', n_jobs = 2, 
                        cat_dissim=categorical_dissimilarity) #gamma= 1, 
    labels= kproto.fit_predict(Season_test, categorical=categorical_columns)

    db_index = davies_bouldin_score(Season_test[collumns_db], labels)

    centroids = pd.DataFrame(kproto.cluster_centroids_, columns=Season_test.columns)
    centroids['cluster'] = centroids.index

    return db_index, labels, centroids

def plot_db(pen =[1,1], nb_samples=200,all_data=False):
    for name,season_df in zip(['Rabi', 'Kharif'],[Rabi, Kharif]):
        DB=[]
        for i in trange(2,15):
            db,_,__ = compute_clusters(season_df, nb_clusters=i,pen =pen, nb_samples=nb_samples,all_data=False)
            DB.append(db)
        plt.plot(np.arange(2,15),DB)
        plt.title(f'Db in function of k for {name}')
        plt.show()



#COMPUTE AND SVAE CLUSTERS 
# print("COMPUTE CLUSTERS RABI")
# db_index_R, labels_R, _ = compute_clusters(Rabi, nb_clusters=9,pen=[1,1],all_data=True)
# print("db_index_R = ",db_index_R)
# databis_R=df_R.copy()
# databis_R['0']= labels_R
# databis_R=databis_R[['0']]
# databis_R.to_csv(f"../../Outputs/Predictions/kproto_labels_Rabi_{date}")

# print("COMPUTE CLUSTERS KHARIF")
# db_index_K, labels_K, _ = compute_clusters(Kharif, nb_clusters=13,pen=[1,1],all_data=True)
# print("db_index_K = ",db_index_K)
# databis_K=df_K.copy()
# databis_K['0']= labels_K
# databis_K=databis_K[['0']]
# databis_K.to_csv(f"../../Outputs/Predictions/kproto_labels_Kharif_{date}")


#FILL 03-pred
def fill_pred(date = "02-03",season = "Rabi"):
    STATES_NOT_INCLUDED = {"Rabi": ['assam', 'uttarakhand', 'jharkhand'], "Kharif":['assam', 'tamil nadu']}

    # Define the root of our project
    # root = "/Users/maximebonnin/Documents/Projects/SCOR/Datathon/"
    # root = "C:/Users/Amaury Lancelin/OneDrive/Main Dossier frr ON/Études/Polytechnique/X 3A/Datathon SCOR/Github (entre lézard)/Datathon/"
    root = "/users/eleves-b/2019/amaury.lancelin/Datathon/"

    # adding roots to the system path
    sys.path.insert(0, root)


    print("Season to be filled...",season)

    STATES_NOT_INCLUDED = STATES_NOT_INCLUDED[season]

    from Environnement.extractClusters import get_closest_keys_scoring, score_fn, get_cluster, get_closest_keys_location

    # Def the path of the clusters file
    pathPreds = root + f"Outputs/Predictions/kproto_labels_{season}_{date}"

    # Define the predictions needed²
    pathSubmissionTranslated = root + f"Data/03_Prediction/GP_Pred_{season}_ID_translated.csv"
    pathSubmission = root + f"Data/03_Prediction/GP_Pred_{season}_ID.csv"

    pathSubFinal = root + f"Outputs/Results/GP_Pred_{season}_{date}.csv"

    df_preds = pd.read_csv(pathPreds)

    df_preds["State"] = df_preds["key"].apply(lambda x: x.split("_")[0])
    df_preds["District"] = df_preds["key"].apply(lambda x: x.split("_")[1])
    df_preds["SubDistrict"] = df_preds["key"].apply(lambda x: x.split("_")[2])
    df_preds["Block"] = df_preds["key"].apply(lambda x: x.split("_")[3])
    df_preds["GP"] = df_preds["key"].apply(lambda x: x.split("_")[4])
    df_preds["Cluster"] = df_preds["0"]

    dfs_preds = {state:df_preds.loc[df_preds["State"]== state] for state in df_preds["State"].unique()}

    df_submission = pd.read_csv(pathSubmission)
    df_sub_translated = pd.read_csv(pathSubmissionTranslated)

    def fill_submission(df_sub_translated, df_submission, dfs_preds, rule="max"):
        """
        Fill the submission with the predictions of the model
        """
        Clusters = [-1]*len(df_submission)
        # print(len(Clusters), len(df_submission))
        for i in trange(len(df_sub_translated)):
            key = df_sub_translated.iloc[i]["key"]
            state = key.split("_")[0]
            if not state in STATES_NOT_INCLUDED:
                # Clusters[i] = get_cluster(get_closest_keys_scoring(key, dfs_preds, score_fn=score_fn), rule=rule)[0]
                Clusters[i] = get_cluster(get_closest_keys_location(key, dfs_preds), rule=rule)[0]

            
            # if i==3:
            #     break

        df_submission["Cluster"] = Clusters

        return df_submission


    df_submission = fill_submission(df_sub_translated, df_submission, dfs_preds, rule="draw")

    df_submission.to_csv(pathSubFinal)


fill_pred(date=date, season='Rabi')
fill_pred(date=date, season='Kharif')