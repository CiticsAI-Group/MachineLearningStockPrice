from train_test import LSTMModel
from train_test import train_all
import time
import os
import numpy as np
from multiprocessing import Process
import Constant
import Result

wind_code_list=Constant.test1[0:350]

def predict(CLASS,predict_day):
    lstm_model = LSTMModel(CLASS,predict_day)
    if (os.path.isfile(lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + Constant.CHECKPOINT)):
        lstm_model.model.load(
            lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + 'tflearn.model.' + str(CLASS))
    else:
        print('No such tflearn.model: '+lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[CLASS] + 'tflearn.model.' + str(CLASS))
    for wind_code in wind_code_list:
        X = lstm_model.generate_predict_X(wind_code)
        if X == []:
            print(wind_code + " is new stock")
        else:
            result = lstm_model.model.predict(X)
            C5_CLASS = np.array(result[0]).argmax()
            C5_PROB = max(result[0])
            real_y = Result.get_y(wind_code)
            real_dis = Result.get_distribution(wind_code)
        print(wind_code)
        print("predict distribution" , result)
        print("real distribution   " , real_dis)
        print("predict y" , C5_CLASS)
        print("real y   " , real_y)


def predict_all(class_list,predict_day_list):
    print ('start prediction ....')
    for predict_day in predict_day_list:
        for i in class_list:
            print('start prediction: class'+ str(i)+' predict day:'+str(predict_day))
            p = Process(target=predict,args=(i,predict_day))
            p.start()
            p.join()
    print('end prediction ....')

if __name__ == "__main__":
    print ('This is main of module "predict.py"')
    class_list = [5]
    predict_day_list=[1]
    predict_all(class_list,predict_day_list)
    print('Finish main process')
