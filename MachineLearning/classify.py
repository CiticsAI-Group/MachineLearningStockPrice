# !usr/bin/env python
# -*- coding:utf-8 -*-

"""
author:simon
file:classify
date:2018/3/26
classify for stock predict.
"""

from pandas import DataFrame
import numpy as np
import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from AI_PreData.TradeData import *
import params


def get_classify(classify,params={}):
    # get classifier.
    if classify == 'SVM':
        # C = 1.0,the C smaller,the generalization ability stronger.
        clf = SVC(**params)
    elif classify == 'KNN':
        # n_neighbors = 5,weights='uniform'/distance
        clf = KNeighborsClassifier(**params)
    elif classify == 'RF':
        # n_estimators = 10, the num of decision tree
        # n_jobs = 1,
        # class_weight = None, a dict param deal with imbalanced dataset.
        # max_depth = None, the max depth of decision tree.
        clf = RandomForestClassifier(**params)
    elif classify == 'GBDT':
        # max_depth :10-100
        # min_samples_split = 2
        # min_samples_split = 1
        # n_estimators = 100
        # learning rate = 1
        # subsample = 1 [0.5-0.8]
        clf = GradientBoostingClassifier(**params)

    return clf


def train_test(classify='GBDT'):
    classifier = get_classify(classify=classify,params=params.PARAMS4GBDT)
    train_X, train_Y, test_X, test_Y = [],[],[],[]
    for stock in get_industry()['626010']:
        x1,y1,x2,y2 = generate_data(wind_code=stock)
    classifier.fit(train_X,train_Y)
    result = classifier.predict(test_X)
    precision,acc = tri_test(result,test_Y)
    print('precision: %f'%precision)
    print('accuracy: %f' %acc)

def tri_test(pre_result,label):
    # return TP,FP,precision 4 tri-clissifier.
    # TP: right predict 4 pos label
    # FP: predict neg label 2 pos.
    # precision: TP/(TP+FP)
    # accuracy: true/all
    TP,FP,TN = 0,0,0
    if len(pre_result)==len(label):
        length = len(pre_result)
        for i in range(length):
            if pre_result[i] == 1 and label[i] == 1:
                TP += 1
            elif pre_result[i] == 1 and label[i] == -1:
                FP += 1
            elif pre_result[i] == -1 and label[i] == -1:
                TN += 1
        return TP/(TP+FP),(TP+TN)/length
        #return TP,TN
    else:
        return 0,0

def generate_data(wind_code,per=0.7):
    dataset = getTradeData(params.start_date, params.end_date, params.fields, params.time_step, params.predict_day, windcodes=wind_code.split())
    X = np.array(dataset[wind_code][0])
    Y = dataset[wind_code][1]
    X = np.reshape(X,(-1,len(params.fields)*params.time_step))
    Y = [class_generator(y) for y in Y]

    cut = int(len(X)*per)
    train_X,train_Y,test_X,test_Y = X[:cut],Y[:cut],X[cut:],Y[cut:]
    return train_X,train_Y,test_X,test_Y

def class_generator(x):
        if x < -0.5: return -1
        if x > 0.5: return 1
        return 0

def get_industry():
    industry_dict = {}
    with open('industry.csv') as f:
        for line in f.readlines():
            lr = [i.strip("\"")for i in line.strip().split(',')]
            if lr[1] in industry_dict:
                industry_dict[lr[1]].append(lr[0])
            else:
                industry_dict[lr[1]] = []
                industry_dict[lr[1]].append(lr[0])
    return industry_dict
