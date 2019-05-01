# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

mlpr = MLPRegressor()
# UpdateTime is an time string
data = pd.read_csv("sample.csv").drop(labels="UpdateTime",axis=1).to_numpy()
labels = pd.read_csv("labels.csv").to_numpy()[:,0]

print(data.shape)
print(labels.shape)
train_data = data[:900]
test_data = data[900:]

train_labels = labels[:900]
test_labels = labels[900:]

mlpr.fit(train_data, train_labels)
print(mlpr.score(train_data,train_labels))
