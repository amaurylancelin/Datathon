# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn 


# %%
# pathData = "RawData/2017/2017_Andhra Pradesh_Kharif.xlsx"
# df=pd.read_excel(pathData)

# %%
#Définition de la fonction add_Loss 
def add_Loss(df,year):
    """return a new_df with a new collumn Loss"""
    Y=np.array([df[f'{y} Yield'] for y in np.arange(year-6,year+1)])
    theta=np.array(df["Indemnity Level"])
    Y=np.partition(Y,2,axis=0)
    threshold=np.mean(Y, axis=0)*theta
    S=np.array(df["Sum Insured (Inr)"])
    L=np.sum(S*np.maximum(np.zeros(Y.shape),threshold-Y)/threshold,axis=0)
    new_df=df
    new_df["Loss"]=L
    return new_df



# %%
def clean_data(df):
    #Suppresion des colonnes sans valeur non nulle
    df = df.drop(columns = ["Block","2000 Yield","2001 Yield","2002 Yield","2003 Yield","2004 Yield","2005 Yield","2017 Yield","2018 Yield"])
    #Suppression des colonnes inutiles
    df = df.drop(columns = ["State","Sub-District","GP"])
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
