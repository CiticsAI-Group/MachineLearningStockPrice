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



lr=0.001
batch_size=50
input_size=9
timestep_size=20
hidden_size=64
layer_num=2
class_num=5


X = tf.placeholder(tf.float32, [None,timestep_size,input_size])
y = tf.placeholder(tf.float32, [None,class_num])
keep_prob=tf.placeholder(tf.float32)

def unit_lstm():
        lstm_cell=rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0,state_is_tuple=True)
        lstm_cell=rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)
        return lstm_cell

mlstm_cell=rnn.MultiRNNCell([unit_lstm()for i in range(3)],state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
outputs, state=tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state,time_major=False)

h_state = state[-1][1]

W=tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1),dtype=tf.float32)
bias=tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
y_pre=tf.nn.softmax(tf.matmul(h_state,W)+bias)

cross_entropy=-tf.reduce_mean(y*tf.log(y_pre))
train_op = tf.train.AdamOpetimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

