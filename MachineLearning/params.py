# !usr/bin/env python
# -*- coding:utf-8 -*-

"""
author:simon
file:params.py
date:2018/3/26
the params 4 classifier.
"""

PARAMS4GBDT = {
    'max_depth' :80,
    'min_samples_split' :4,
    'min_samples_leaf':3,
    'n_estimators':100,
    'learning_rate':0.2,
    'subsample':0.7
}

start_date = 20160101
end_date = 20180322
fields = ['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_VOLUME', 'S_DQ_PCTCHANGE', 'S_DQ_TRADESTATUS', 'MONTH', 'WEEK']
time_step = 7
predict_day = 1