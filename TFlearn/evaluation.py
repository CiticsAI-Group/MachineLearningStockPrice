from train_test import LSTMModel
from train_test import train_all
import time
import os
import numpy as np
from multiprocessing import Process
import Constant
import Result

wind_code_list=Constant.test1[0:350]
#wind_code_list=["600000.SH","600764.SH","000680.SZ"]

def predict(wind_code_list):
    lstm_model = LSTMModel(5,1)
    lstm_model.model.load(
            lstm_model.CHECK_POINT_PATH + Constant.CLASS_2_PATH_DICT[5] + 'tflearn.model.' + str(5))
    eva=[]
    for wind_code in wind_code_list:
        X = lstm_model.generate_predict_X(wind_code)
        result = lstm_model.model.predict(X)
        C5_CLASS = np.array(result[0]).argmax()
        real_y = Result.get_y(wind_code)
        if C5_CLASS == real_y:
            t1=1
        else:
            t1=0
        if C5_CLASS > 2 and real_y >=2:
            t2=1
        elif C5_CLASS < 2 and real_y <=2:
            t2=1
        else:
            t2=0
        e = [C5_CLASS,real_y,t1,t2]
        eva.append(e)
    return eva

evaluation=predict(wind_code_list)
print(evaluation)
acc1=0
acc2=0
for i in range(len(evaluation)):
    acc1=acc1+evaluation[i][2]
    acc2=acc2+evaluation[i][3]
    accr1=acc1/len(evaluation)
    accr2=acc2/len(evaluation)
print(acc1,acc2,accr1,accr2)



