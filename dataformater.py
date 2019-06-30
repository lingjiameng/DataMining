#-*-coding:utf-8 -*-

import argparse

import pandas as pd
 
parser = argparse.ArgumentParser("处理数据生成label")

parser.add_argument("--input",default="",type=str,help="输入的文件")
parser.add_argument("--out",default="" ,type=str, help="输出的文件")

if __name__ == "__main__":
    args = parser.parse_args()

    STEP = 1  

    in_file = args.input
    out_file = args.out

    sample = pd.read_csv(in_file)

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
    labels.to_csv(out_file, index=False)

    # 数据和对应label合并
    # data_with_label = pd.concat([sample,labels],axis=1) 

    # print(data_with_label.columns)
