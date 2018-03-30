from __future__ import print_function
from AI_Data.dataSource import DataSource
import pandas as pd
import numpy as np
from AI_PreData.TradeData import *
data_source = DataSource()

fields = ['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_VOLUME', 'S_DQ_PCTCHANGE', 'S_DQ_TRADESTATUS', 'MONTH', 'WEEK']

def class_generator(x):
    if x < -1.5: return 0
    if x < -0.25: return 1
    if x < 0.25: return 2
    if x < 1.5: return 3
    return 4


def get_y(wind_code):
    data = getOneDay(20160101, 20180315, fields, 40, 1, windcodes=wind_code.split())
    y = data[wind_code][1]
    c = class_generator(y)
    return c


def get_distribution(wind_code):
    data = getTradeData(20160101, 20180315, fields, 2, 1, windcodes=wind_code.split())
    Y = data[wind_code][1]
    c = [class_generator(y) for y in Y]
    c0 = c.count(0) / len(c)
    c1 = c.count(1) / len(c)
    c2 = c.count(2) / len(c)
    c3 = c.count(3) / len(c)
    c4 = c.count(4) / len(c)
    return [c0, c1, c2, c3, c4]


