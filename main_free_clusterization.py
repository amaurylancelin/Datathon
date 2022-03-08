#IMPORTS
import argparse
from pathlib import Path
import datetime as dt
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import warnings
warnings.filterwarnings("ignore")

#Our own libraries
from src.clean import clean_data, regroup_crop
from src.utils import add_crop_categories, add_climate_clusters, add_Loss


# ----------------------------------------------------------------------------------------------------------------------

date_today = dt.datetime.now().strftime("%Y-%m-%d")

argparser = argparse.ArgumentParser(description="Compute the clusterization with k-means or k-prototypes")
argparser.add_argument("--season", type=str, help="Season to be filled", required=True)
argparser.add_argument("--name_id", type=str, default=str(date_today), help="Id to be added at the end of the file name")
argparser.add_argument("--output_dir", type=str, default="output/clusters", help="Output directory")
argparser.add_argument("--algo", type=str, default='kmeans', help="Choice of algorithm to use: 'kmeans' or 'kproto'.", required=True)
argparser.add_argument("--k", type=int, default=8, help="Number of clusters", required=True)
argparser.add_argument("--pen", type=float, nargs="+", default=[1.,1.],
                       help="""Penalisation in the k-prototypes algorithm : [pen_climate_clusters, pen_crop_categories]""", required=True)
args = argparser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Retrieve all args
    season = args.season
    name_id = args.name_id
    output_dir = args.output_dir
    algo = args.algo
    k = args.k
    pen = args.pen
    
    if len(pen) != 2:
        raise ValueError("Wrong number of parameters")
    

    # DATA PREPROCESSING
    # Path to the dataset
    year = 2019
    pathData= f"data/merged_data/RawData_{year}_{season}.csv"
    df= pd.read_csv(pathData)

    # Clean an add loss to data
    df=add_Loss(clean_data(df))

    # Normalise the data
    data=df.copy(deep=True)
    scale = StandardScaler()
    data.loc[:,data.columns !='Crop']=scale.fit_transform(data.loc[:,data.columns !='Crop'])

    # Data for Davis_Bouldin criteria 
    collumns_db = [f'Lp_{i}' for i in range(2011,2018)]
    data_db=data[collumns_db]

    if algo == 'kproto' :
        # DATA PREPROCESSING SPECIFIC FOR K-PROTOTYPES

        def find_state(key) : 
            return key.split('_')[0]
        # Select the collumns of interest

        collumns_db= [f'Lp_{i}' for i in range(2011,2018)] 
        collumns = collumns_db + ['Crop']
        
        data_kproto = regroup_crop(data)
        data_kproto = data_kproto[collumns]
        data_kproto = data_kproto.reset_index()
        data_kproto['State']=(data_kproto.reset_index())['key'].map(find_state)
        data_kproto= data_kproto.set_index('key')
        data_kproto = add_climate_clusters(data_kproto, season)
        data_kproto = add_crop_categories(data_kproto, season).drop(columns=['Crop', 'State'])
        

        # Functions for computing clusters
        def cat_diss_pen(pen) :
            def categorical_dissimilarity(a, b, **_):
                """Dissimilarity function with penalization"""
                return (a[:,0] != b[0])*pen[0] + (a[:,1] != b[1])*pen[1]
            return categorical_dissimilarity
        
        def kproto_clusters(data, nb_clusters=7, pen =[1,1]):
            """Compute clusters for k-prototypes algorithm. Return the Davies-Bouldin index, labels, and the centroids"""
            
            #'7' -> 'climate_clusters' and '8' -> 'crop_categories'
            categorical_columns = [7,8]
            categorical_dissimilarity = cat_diss_pen(pen)
            kproto = KPrototypes(n_clusters= nb_clusters, init='Cao', n_jobs = 2, 
                                cat_dissim=categorical_dissimilarity, random_state=0)
            labels= kproto.fit_predict(data, categorical=categorical_columns)
            db_index = davies_bouldin_score(data[collumns_db], labels)
            centroids = pd.DataFrame(kproto.cluster_centroids_, columns=data.columns)
            centroids['cluster'] = centroids.index
            return db_index, labels, centroids

        # COMPUTE AND SAVE CLUSTERS 
        print(f"Compute clusters for {season}, id =  {name_id}, with {algo}. Number of clusters k = {k}")
        db_index, labels, _ = kproto_clusters(data_kproto, nb_clusters=k,pen=pen)
        print(f"db index for {season} with k = {k} : ", db_index)
        databis=df.copy()
        databis['Label']= labels
        databis=databis[['Label']]
        databis.to_csv(Path(output_dir) / f"kproto_labels_{season}_{name_id}.csv")



    elif algo == "kmeans":
        # COMPUTE AND SAVE CLUSTERS
        print(f"Compute clusters for {season}, id =  {name_id}, with {algo}. Number of clusters k = {k}")
        kmeans= KMeans(init="k-means++", n_clusters=k, max_iter=500, n_init=15, random_state=0).fit(data_db)
        labels= kmeans.labels_
        db_index = davies_bouldin_score(data_db, labels)
        print(f"db index for {season} with k = {k} : ", db_index)
        databis=df.copy()
        databis['Label']= labels
        databis=databis[['Label']]
        databis.to_csv(Path(output_dir) / f"kmeans_labels_{season}_{name_id}.csv")
        
        
    else : 
        raise ValueError("Invalid algorithm, use instead 'kmeans' or 'kproto'.")


if __name__ == "__main__":
    main()