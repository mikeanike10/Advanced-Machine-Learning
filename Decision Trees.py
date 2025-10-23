#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:57:03 2023

@author: Michael Glass
"""

# INTENT: System, graphics, file locations prepared AND save_fig() defined 

# Implementation of A. Geron, edited by Eric Braude

# Common imports
import numpy as np
import os # functions for interacting portably with OS, e.g., file system

# With respect to randomness, this notebook's output will be same across runs
np.random.seed(42) # "42" is arbitrary (but unchanging)


#task imports
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

#data import
df=pd.read_csv("drug200.csv")

#data cleaning

df["Cholesterol"]=np.where(df["Cholesterol"]=="HIGH",1,0)
df["Sex"]=np.where(df["Sex"]=="M",1,0)
df["BP"]=df["BP"].replace(["HIGH","LOW","NORMAL"],[2,0,1])
df["Drug"]=df["Drug"].replace(['drugY',"drugC",'drugX','drugA','drugB'],[0,1,2,3,4])
x=df[df.columns[0:5]].to_numpy()
y=df[df.columns[5]].to_numpy()

#setting up decision tree classifier 
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42,criterion='gini') # definition

#fitting
tree_clf.fit(x, y) 

#predictions 
#1
tree_clf.predict([[23,1,1,0,8.171]]) #array([2])

#2
tree_clf.predict([[73,1,2,1,31.924]]) #array([0])

#3
tree_clf.predict([[43,0,2,0,14.329]]) #array([3])
    

#altered data
df["Na_to_K"].mean()
df.groupby("Drug")["Na_to_K"].mean()
df.groupby("Drug")["Age"].mean()
df.groupby("Drug")["Na_to_K"].median()
df.groupby("Drug")["Age"].median()
df.groupby("Drug")["Na_to_K"].max()
df.groupby("Drug")["Age"].max()
df.groupby("Drug")["Na_to_K"].min()
df.groupby("Drug")["Age"].min()
df.groupby("Drug")["Na_to_K"].count()
plt.hist(df["Age"]) 

dfsub=df[df["Na_to_K"]<=df["Na_to_K"].median()]
dfsub.groupby("Drug")["Na_to_K"].count()
dfsub.groupby("Drug")["Na_to_K"].mean()
dfsub.groupby("Drug")["Na_to_K"].max()
x_tweaked=dfsub[dfsub.columns[0:5]].to_numpy()
y_tweaked=dfsub[dfsub.columns[5]].to_numpy()


#fit
tree_clf.fit(x_tweaked,y_tweaked)

#predictions 
#1
tree_clf.predict([[23,1,1,0,8.171]]) #array([2])

#2
tree_clf.predict([[73,1,2,1,31.924]]) #array([4])

#3
tree_clf.predict([[43,0,2,0,14.329]]) #array([3])

#adding inconsitencies 
for i in range(4):
    df.loc[len(df.index)] = [161,0,2,1,3.555,1]
    df.loc[len(df.index)] = [23,2,1,1,130.093,3]
    df.loc[len(df.index)] = [47,1,3,1,114,3]
    df.loc[len(df.index)] = [28,0,0,4,68.607,4]
    df.loc[len(df.index)] = [609,1,1,2,50.376,2]   
    df.loc[len(df.index)] = [161,0,2,1,123.555,1]
    df.loc[len(df.index)] = [23,2,1,1,130.093,4]
    df.loc[len(df.index)] = [387,1,3,1,114,3]
    df.loc[len(df.index)] = [28,0,0,4,68.607,2]
    df.loc[len(df.index)] = [60,1,1,2,50.376,3] 
    df.loc[len(df.index)] = [409,0,2,1,123.555,1]
    df.loc[len(df.index)] = [234,2,1,1,130.093,1]
    df.loc[len(df.index)] = [47,1,3,1,114,3]
    df.loc[len(df.index)] = [28,0,0,4,3.607,3]
    df.loc[len(df.index)] = [60,1,1,2,50.376,2] 
    
#fitting 
x=df[df.columns[0:5]].to_numpy()
y=df[df.columns[5]].to_numpy()
tree_clf.fit(x, y) 

#predictions 
#1
tree_clf.predict([[23,1,1,0,8.171]]) #array([2])

#2
tree_clf.predict([[73,1,2,1,31.924]]) #array([0])

#3
tree_clf.predict([[43,0,2,0,14.329]]) #array([3])
    
    
    
    
    
    
    
    
    
    
