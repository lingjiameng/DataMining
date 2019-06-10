# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

# load data
# UpdateTime is an time string
data = pd.read_csv("sample.csv").drop(labels="UpdateTime",axis=1).to_numpy()
labels = pd.read_csv("labels.csv").to_numpy()[:,0]

mlpr = MLPRegressor(hidden_layer =(10,),
                    activation = "logistic",
                    batch_size = 1000,
                    alpha=0.0001,
                    learning_rate='adaptive',
                    )

mlpr.fit(data,labels)

