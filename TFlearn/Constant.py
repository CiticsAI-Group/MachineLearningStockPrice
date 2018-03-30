from __future__ import print_function
from AI_Data.dataSource import DataSource
import os
import pandas as pd
import numpy as np
import test
from AI_PreData.TradeData import *

data_source = DataSource()

###### FILE PATH ######
TFLEARN_FILE = '/home/ubuntu2/PycharmProjects/AI-Research/TFlearn'
CHECK_POINT_PATH = 'checkpoint/'
CHECKPOINT='checkpoint'
CLASS_2_PATH_DICT={2:"C2/",5:"C5/", 3:"C3/", 7: 'C7/'}

###### INDUSTRY CODE ######
indset = pd.read_csv('industry.csv')
ind_list=list(indset['IND_CODE'].unique())

for i in range(len(ind_list)):
    ind_list[i]=str(ind_list[i])     ##list of ind code

def code_by_ind(indcode):
    code_list=list(indset[indset.IND_CODE == int(indcode)].WIND_CODE)
    return code_list                 ##list of wind code under each industry class
#print(code_by_ind(ind_list[0]))



###### TRAIN MODEL ######
start_date = 20160101
end_date = 20180322
fields = ['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_VOLUME', 'S_DQ_PCTCHANGE', 'S_DQ_TRADESTATUS', 'MONTH', 'WEEK']


test0= list(set(code_by_ind(ind_list[0])).intersection(test.code_all))
test1= list(set(code_by_ind(ind_list[1])).intersection(test.code_all))




