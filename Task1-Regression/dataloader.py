#-*-coding:utf-8 -*-
import pandas as pd

STEP = 1  

sample = pd.read_csv("sample10000.csv")

# print(sample.columns)
# print(sample.describe())

# label equals (the future n-th tick's AskPrice1 + the
# future n-th tick's BidPrice1 - the current tick's AskPrice1 - the current tick's BidPrice1) / 2

# 获取价格
prices = sample.loc[:,("AskPrice1","BidPrice1")]
# print(prices.columns)
# print(prices.describe())

# 计算标签
labels = (prices.shift(-1*STEP) - prices).sum(axis=1) #绝大多数为0
# labels.columns = ["label"]
labels =labels.to_frame(name='label')

#标签保存
labels.to_csv("labels10000.csv",index=False)

# 数据和对应label合并
# data_with_label = pd.concat([sample,labels],axis=1) 

# print(data_with_label.columns)
