# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 23:02:05 2018

@author: Simon
"""

#特征保存

import pickle

data = (x_train, y_train, x_test)

fp = open('./data_tfidf.pkl', 'wb')

pickle.dump(data, fp)

fp.close()

#特征读取

import pickle

features_path = '*****.pkl'

fp = open(features_path, 'rb')

x_train, y_train, x_test = pickle.load(fp)

fp.close()

#模型保存

from sklearn.externals import joblib

joblib.dump(lin_clf, "linearsvm_model_Tfid_1.m")

#模型读取

from sklearn.externals import joblib

svm = joblib.load("linearsvm_model_Tfid_1.m")