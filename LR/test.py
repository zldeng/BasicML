#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-23 15:34:25
'''
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from LogisticRegression import LogisticRegression


np.random.seed(12)

def generateData(data_cnt):
	x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], data_cnt)
	x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], data_cnt)
		
	simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
	simulated_labels = np.hstack((np.zeros(data_cnt),np.ones(data_cnt)))
	
	return simulated_separableish_features,simulated_labels

train_data_cnt = 5000
train_x,train_y = generateData(train_data_cnt)

print train_x.shape,train_y.shape

test_data_cnt = 1000
test_x,test_y = generateData(test_data_cnt)

lr = LogisticRegression()
lr.fit(train_x,train_y)


pred_res = lr.predict(test_x)

print pred_res[:10]

correct_cnt = np.sum((pred_res == test_y).astype(int))

print correct_cnt

print correct_cnt * 1.0/test_x.shape[0]
