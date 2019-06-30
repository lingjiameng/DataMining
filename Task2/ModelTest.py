##### Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Task1-Regression'))
	print(os.getcwd())
except:
	pass

#####
#####
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


#####
##### hyperparameter
TRAIN_SIZE = 9000
# UpdateTime is an time string


#####
##### get tain_data and test_data
data = pd.read_csv("sample10000.csv").drop(labels="UpdateTime",axis=1).to_numpy()
labels = pd.read_csv("labels10000.csv").to_numpy()[:,0]

train_data = data[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:]

train_labels = labels[:TRAIN_SIZE]
test_labels = labels[TRAIN_SIZE:]


#####
def model_score(model):
    ##### predict and score
    err = (test_labels -  model.predict(test_data))
    sqerr = (err*err).sum()
    mse = sqerr*1.0/len(err)
    # print(regr.get_params())
    #print("OOB score",regr.oob_score_)
    print("R^2 for train data :",model.score(train_data,train_labels))
    print("R^2 for test data :",model.score(test_data, test_labels))
    print("mean squre error:",mse)


#####
##### define model and train
def test_RF():
    regr = RandomForestRegressor(n_estimators=200,oob_score=True)
    regr.fit(train_data,train_labels)
    model_score(regr)
    # R^2 for train data : 0.8658618241928562
    # R^2 for test data : -0.07898389513463266
    # squre error: 313.67217500000004


#####
##### define DNN(MLPregressor)
# R^2 for train data : -2.4562869375355234e-05
# R^2 for test data : -0.0005819326087599386
# squre error: 290.8798847529891
def test_MLP():
    mlpr = MLPRegressor(hidden_layer_sizes=(10),
                        activation="logistic",batch_size=1000)
    # 激活函数很关键，relu 和 identity 基本不工作
    # batch_size 影响也很大
    mlpr.fit(train_data,train_labels)
    model_score(mlpr)


#####
def find_MLP():
    mlp_params = {
        "hidden_layer_sizes": [(x,) for x in range(10,150,10)],
        "activation":["logistic"],
        "batch_size":[1000,],
        "learning_rate":["adaptive"]
    }
    mlpr = MLPRegressor()
    clf = GridSearchCV(mlpr,mlp_params,cv=10)
    clf.fit(data,labels)
    pprint.pprint(clf.cv_results_)
    pprint.pprint(clf.best_params_)


#####
#     R^2 for train data : 0.31327344425793446
#     R^2 for test data : 0.04436323074152071
#     squre error: 277.81384437241996

def test_GBR():
    gbr = GradientBoostingRegressor()
    gbr.fit(train_data,train_labels)
    model_score(gbr)


#####
# R^2 for train data : 0.0993108628821433
# R^2 for test data : 0.08070103797930306
# squre error: 267.2500546046555
def test_ABR():
    abr = AdaBoostRegressor(learning_rate=0.01)
    abr.fit(train_data,train_labels)
    model_score(abr)
    


#####

test_RF()
test_MLP()
test_GBR()
test_ABR()


mypca = TSNE(n_components=2, init="pca")
new_data = mypca.fit_transform(np_data)

train_data = new_data[:TRAIN_SIZE]
test_data = new_data[TRAIN_SIZE:]


test_RF()
test_MLP()
test_GBR()
test_ABR()
#####



