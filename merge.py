import pandas as pd
import numpy as np

from tqdm import trange


columns_to_keep = ['Season',
       'Crop', 'Area Sown (Ha)', 'Area Insured (Ha)', 'SI Per Ha (Inr/Ha)',
       'Sum Insured (Inr)', 'Indemnity Level', 'key_str', 'Loss']

yields = ['State', 'Cluster', 'District', 'Sub-District', 'Block', 'GP', 'Season',
        '2000 Yield', '2001 Yield', '2002 Yield', '2003 Yield', '2004 Yield', 
        '2005 Yield', '2006 Yield', '2007 Yield', '2008 Yield', '2009 Yield', 
        '2010 Yield', '2011 Yield', '2012 Yield', '2013 Yield', '2014 Yield', 
        '2015 Yield', '2016 Yield', '2017 Yield', '2018 Yield', 'key_str'
]

def add_key_str(df):

    df["GP"] = df["GP"].fillna("")
    df["Block"] = df["Block"].fillna("").astype(str)
    

    df["key_str"] = df["State"].astype(str) + "_" + df["District"].astype(str)
    df["key_str"] += "_" + df["Sub-District"].astype(str) + "_"
    df["key_str"] += df["Block"].astype(str) + "_"
    df["key_str"] += df["GP"].astype(str)

    df["key_str"] = df["key_str"].str.lower()

    return df

def merge_year(dfs):
    df_merged = dfs[0].copy()[columns_to_keep]
    df_merged = df_merged.merge(dfs[1][columns_to_keep], on=["Season", "key_str"],suffixes=('_2017', '_2018'))
    df_merged = df_merged.merge(dfs[2][columns_to_keep], on=["Season", "key_str"],suffixes=("_2018", '_2019'))

    return df_merged