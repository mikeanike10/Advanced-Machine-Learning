#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:33:41 2023

@author: Michael Glass
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import bnlearn as bn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("mushrooms.csv")
data.describe
data.info

# Pre-processing of the input dataset
dfhot, dfnum = bn.df2onehot(data)

# Structure learning
DAG = bn.structure_learning.fit(dfnum)

# Plot
G = bn.plot(DAG)

# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)
model

#class 0 is e, class 1 is p
#make inference example 1
q = bn.inference.fit(model, variables=['class'], evidence={"cap-shape":1
                                                        })
print(q.df)

#make inference example 2
q = bn.inference.fit(model, variables=['class'], evidence={"cap-shape":1,
                                                           "cap-surface":3,
                                                           "cap-color":4,
                                                           "bruises":0,
                                                           "odor":4
                                                        })
print(q.df)
