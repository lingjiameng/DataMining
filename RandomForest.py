#%%
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor

#%% hyperparameter
TRAIN_SIZE = 9000
# UpdateTime is an time string

#%% get tain_data and test_data
data = pd.read_csv("sample10000.csv").drop(labels="UpdateTime",axis=1).to_numpy()
labels = pd.read_csv("labels10000.csv").to_numpy()[:,0]

train_data = data[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:]

train_labels = labels[:TRAIN_SIZE]
test_labels = labels[TRAIN_SIZE:]

#%% define model and train
regr = RandomForestRegressor(n_estimators=200,oob_score=True)
regr.fit(train_data,train_labels)


#%% predict and score
err = (test_labels -  regr.predict(test_data))
sqerr = (err*err).sum()
# print(regr.get_params())
print("OOB score",regr.oob_score_)
print("R^2 for train data :",regr.score(train_data,train_labels))
print("R^2 for test data :",regr.score(test_data, test_labels))
print("squre error:",sqerr)
# {'bootstrap': True, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
#     'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
# 0.8549146826557474
# -0.512199830508475
# 36.0484
