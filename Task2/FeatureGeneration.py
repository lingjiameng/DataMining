#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Task2'))
	print(os.getcwd())
except:
	pass

#%%
import pprint
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#%%
#%% hyperparameter
TRAIN_SIZE = 9000
# UpdateTime is an time string


#%%
#%% get tain_data and test_data
np_data = pd.read_csv("sample10000.csv").drop(labels="UpdateTime",axis=1).to_numpy()
np_labels = pd.read_csv("labels10000.csv").to_numpy()[:,0]

train_data = np_data[:TRAIN_SIZE]
test_data = np_data[TRAIN_SIZE:]

train_labels = np_labels[:TRAIN_SIZE]
test_labels = np_labels[TRAIN_SIZE:]


def model_score(model):
    #%% predict and score
    err = (test_labels - model.predict(test_data))
    sqerr = (err*err).sum()
    mse = sqerr*1.0/len(err)
    # print(regr.get_params())
    #print("OOB score",regr.oob_score_)
    print("R^2 for train data :", model.score(train_data, train_labels))
    print("R^2 for test data :", model.score(test_data, test_labels))
    print("mean squre error:", mse)




#%% [markdown]
# ## PCA
mypca = TSNE(n_components=2,init="pca")
new_data = mypca.fit_transform(np_data)

train_data = new_data[:TRAIN_SIZE]
test_data = new_data[TRAIN_SIZE:]

abr = AdaBoostRegressor(learning_rate=0.01)
abr.fit(train_data, train_labels)
model_score(abr)
