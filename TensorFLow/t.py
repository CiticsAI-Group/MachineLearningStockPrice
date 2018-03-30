#coding:utf-8
from __future__ import print_function
from AI_Data.dataSource import DataSource
import os
import time
import Constant
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tflearn.data_utils import to_categorical
from multiprocessing import Process
from AI_PreData.TradeData import *
data_source = DataSource()

######### PARAMETERS ########

time_step = 60      #时间步
batch_size = 50     #每一批次训练多少个样例
INPUT_SIZE = 9      #输入层维度
OUTPUT_SIZE = 3     #输出层维度
CELL_SIZE = 256     #hidden unit size
lr = 0.01           #学习率
layers=2

predict_day=1
classes=3

######## IMPORT DATA ########
def class_generator(x, c):
    if (c == 5):
        if x < -1.5: return 0
        if x < -0.25: return 1
        if x < 0.25: return 2
        if x < 1.5: return 3
        return 4
    if (c == 3):
        if x < -0.1: return 0
        if x > 0.1: return 2
        return 1

def generate_data(wind_code):
    dataset = getTradeData(Constant.start_date, Constant.end_date, Constant.fields, time_step, predict_day, windcodes=wind_code.split())
    X = dataset[wind_code][0]
    Y = dataset[wind_code][1]
    Y = to_categorical([class_generator(y, classes) for y in Y], nb_classes=classes)
    Y = Y.tolist()
    return X, Y

wind_code_list=Constant.test0
print(wind_code_list,len(wind_code_list))

train_x=[]
train_y=[]
for wind_code in wind_code_list:
    x,y = generate_data(wind_code)
    train_x.extend(x)
    train_y.extend(y)
print(train_x)


batch_index=[]
for i in range(len(train_y)):
    if i % batch_size == 0:
        batch_index.append(i)

print(batch_index)
