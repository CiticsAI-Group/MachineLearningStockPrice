from __future__ import print_function
from AI_Data.dataSource import DataSource
import numpy as np
import pandas as pd
import os
import time
import tflearn
import tensorflow as tf
import Constant
from tflearn.data_utils import to_categorical
from multiprocessing import Process
from AI_PreData.TradeData import *

data_source = DataSource()




wind_code_list=Constant.test1[0:350]
###### LSTM MODEL ######
class LSTMModel:
    timesteps = 40
    hidden_layers = 128
    global features
    global classes
    global model
    global predict_day
    SUB_PATH = ''
    PATH_TO_FILE = ''
    CHECK_POINT_PATH = ''
    BEST_CHECK_POINT_PATH = ''

    def __init__(self, classes, predict_day):
        self.classes = classes
        self.predict_day = predict_day
        self.SUB_PATH = 'T' + str(self.predict_day) + '/'
        self.PATH_TO_FILE = Constant.TFLEARN_FILE + self.SUB_PATH
        self.CHECK_POINT_PATH = self.PATH_TO_FILE + Constant.CHECK_POINT_PATH
        self.features = len(Constant.fields)
        self.date = time.strftime("%Y%m%d", time.localtime())
        self.model = self.get_model()

    def generate_data(self, wind_code):
        dataset = getTradeData(Constant.start_date, Constant.end_date, Constant.fields, self.timesteps, self.predict_day, windcodes = wind_code.split())
        X = dataset[wind_code][0]
        X = np.array(X, dtype=np.float32)
        Y = dataset[wind_code][1]
        Y = to_categorical([self.class_generator(y, self.classes) for y in Y], nb_classes=self.classes)
        return X, Y


    def generate_predict_X(self, wind_code):
        dataset = getPredictData(self.timesteps, Constant.fields, windcodes=wind_code.split())
        X = dataset[wind_code]
        X = [X]
        X = np.array(X, dtype=np.float32)
        return X

    def class_generator(self, x, c):
        if (c == 5):
            if x < -1.5: return 0
            if x < -0.25: return 1
            if x < 0.25: return 2
            if x < 1.5: return 3
            return 4


    def new_data(self, wind_code, isNew):
        if isNew:
            dataset = getTradeData(Constant.start_date, Constant.end_date, Constant.fields, self.timesteps, self.predict_day,
                                   windcodes=wind_code.split())
            X = dataset[wind_code][0]
            X = np.array(X, dtype=np.float32)
            Y = dataset[wind_code][1]
            Y = to_categorical([self.class_generator(y, self.classes) for y in Y], nb_classes=self.classes)
            return X, Y
        else:
            dataset = getOneDay(Constant.start_date, int(self.date), Constant.fields, self.timesteps, self.predict_day,
                                windcodes=wind_code.split())
            X = dataset[wind_code][0]
            X = [X]
            X = np.array(X, dtype=np.float32)
            Y = dataset[wind_code][1]
            Y = to_categorical([self.class_generator(Y, self.classes)], nb_classes=self.classes)
            return X, Y


    def get_model(self):
        net = tflearn.input_data([None, self.timesteps, self.features])
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8,return_seq=True)
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8, return_seq=True)
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8)
        net = tflearn.fully_connected(net, self.classes, activation='softmax')
        net = tflearn.regression(net, optimizer='Adam', learning_rate=0.0001, loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model


    def splitDataset(self, X, Y, train_validation_ratio):
        train_data_len = int(X.__len__() * train_validation_ratio)
        trainX = X[:train_data_len]
        trainY = Y[:train_data_len]
        validationX = X[train_data_len:]
        validationY = Y[train_data_len:]
        return trainX, trainY, validationX, validationY

    def train_model(self, X, Y, train_validation_ratio, n_epoch, batch_size):
        if(X.__len__()==0):
            print('No Training data. return')
            return
        if (train_validation_ratio == 1):
            self.model.fit(X, Y, show_metric=True,
                           batch_size=batch_size, n_epoch=n_epoch)
        else:
            trainX, trainY, validationX, validationY = self.splitDataset(X, Y, train_validation_ratio)
            self.model.fit(trainX, trainY, validation_set=(validationX, validationY), show_metric=True,
                           batch_size=batch_size, n_epoch=n_epoch)
        return self.model



###### TRAIN MODEL ######
def train_model(CLASS,predict_day,FORCE=False):
    MAX_DATASET_SIZE = 100000
    BATCH_SIZE = 1000
    RATIO = 1
    EPOCH = 5
    lstm_model = LSTMModel(CLASS,predict_day)
    print(lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + Constant.CHECKPOINT)
    isNew = True
    if (os.path.isfile(lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + Constant.CHECKPOINT)):
        lstm_model.model.load(
            lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + 'tflearn.model.' + str(CLASS))
        isNew = False
    if (FORCE == True):
        isNew = True
    X = []
    Y = []
    for i in range(EPOCH):
        for wind_code in wind_code_list:
            global tempX
            global tempY
            if (isNew):
                tempX, tempY = lstm_model.generate_data(wind_code)
            else:
                tempX, tempY = lstm_model.new_data(wind_code, isNew)
            if (tempY == []):
                print(wind_code + " is new stock and less than 50 Trade Day !")
            else:
                X.extend(tempX)
                Y.extend(tempY)
            print("train model using " + wind_code + " total size=" + Y.__len__().__str__())
            if (Y.__len__() > MAX_DATASET_SIZE):
                lstm_model.train_model(X, Y, RATIO, 1, BATCH_SIZE)
                X.clear()
                Y.clear()
        lstm_model.train_model(X, Y, RATIO, 1, BATCH_SIZE)
        X.clear()
        Y.clear()
        print("Finish EPOCH" + str(i))
    lstm_model.model.save(lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + 'tflearn.model.' + str(CLASS))

def train_all(class_list,predict_day_list,FORCE=False):
    print ('start prediction ....')
    for predict_day in predict_day_list:
        for i in class_list:
            p = Process(target=train_model,args=(i,predict_day,FORCE))
            p.start()
            p.join()
    print('end prediction ....')

if __name__ == "__main__":
    print ('This is main of module "train.py"')
    class_list = [5]
    predict_day_list = [1]
    train_all(class_list,predict_day_list)
    print('Finish main process')
