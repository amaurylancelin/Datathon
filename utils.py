# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn 


# %%
pathData = "RawData/2017/2017_Andhra Pradesh_Kharif.xlsx"
df=pd.read_excel(pathData)


# %%
#Définition de la fonction add_Loss 
def add_Loss(df,year):
    """return a new_df with a new collumn Loss"""
    Y=np.array([df[f'{YEAR} Yield'] for YEAR in np.arange(year-6,year+1)])
    theta=np.array(df["Indemnity Level"])
    index=np.argpartition(Y,2,axis=0)
    Y=Y[index[2:]]
    Y=Y[:,:,0]
    threshold=np.mean(Y, axis=0)*theta
    S=np.array(df["Sum Insured (Inr)"])
    L=np.sum(S*np.maximum(np.zeros(Y.shape),threshold-Y)/threshold,axis=0)
    new_df=df
    new_df["Loss"]=L
    return new_df


