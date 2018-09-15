#!/usr/bin/python
# -*- coding: UTF-8 -*-

print ("----------程序开始运行！！！------------")
import pickle
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import sys,csv
import time
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import  train_test_split
#from sklearn.neural_network import MLPClassifier
start_time = time.time()

path='E:/Heitao/达观杯数据集/new_data/'

maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

print('1  read data begin')
read_start_time = time.time()
df_train=pd.read_csv(path + '/train_set.csv',engine='python',encoding='gbk')
df_test=pd.read_csv(path + '/test_set.csv',engine='python',encoding='gbk')
print (df_train.shape)
#df_train.drop(columns=['id','article'],inplace=True)
#df_test.drop(columns=['article'],inplace=True)
# df_train.drop(columns=['id'],inplace=True)
df_train["word_seg"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
df_test["word_seg"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)
read_end_time = time.time()
read_druing_time = read_end_time - read_start_time
print ("读取+预处理数据结束:耗时%s"%read_druing_time)


print('2  vertorizer data')
vert_start_time = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1
xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size=0.30, random_state=531)
vert_end_time = time.time()
vert_druing_time = vert_end_time - vert_start_time
print ("特征工程结束:耗时%s"%vert_druing_time)


print('3  train model')
fit_start_time = time.time()
#lg = LogisticRegression(C=120,dual=True)
#lg.fit(xTrain,yTrain)
clf = svm.LinearSVC(C=6,dual=False)
#clf =  MLPClassifier(solver='adam', alpha=0.01,hidden_layer_sizes=(5, 5), random_state=1, activation='relu', learning_rate_init=0.01,max_iter=1000)
clf.fit(x_train,y_train)
#Accumulate auc on test set
prediction = clf.predict(xTest)
prediction2 = clf.predict(x_train)
correct = accuracy_score(yTest, prediction)
correct2 = accuracy_score(y_train, prediction2)
print(correct)
print(correct2)
fit_end_time = time.time()
fit_druing_time = fit_end_time - fit_start_time
print ("数据训练结束:耗时%s"%fit_druing_time)


print('4  save model')
#joblib.dump(sgd, "sgd_model_Tfid_1.m")
y_test= clf.predict(x_test)
df_test['class']=y_test.tolist()
df_test['class']=df_test['class']+1
df_result = df_test.loc[:,['id','class']]

print('5  save predictable data')
df_result.to_csv(path + 'result_svm(c6).csv',index=False)


# 保存特征
with open('./y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
with open('./x_train.pickle', 'wb') as f:
    pickle.dump(x_train, f)
with open('./x_test.pickle', 'wb') as f:
    pickle.dump(x_test, f)

# 读取特征
# print('读取 feature:')
# with open('./feature/tfidf/x_train.pickle', 'rb') as f:
#     x_train = pickle.load(f)




# 保存模型
joblib.dump(clf, "svm(C6).pkl")
# 读取模型
print('读取* model')
my_model_loaded = joblib.load("LinearSVC(C4).pkl")


'''程序运行结束'''
end_time = time.time()
during_time = end_time-start_time
print("--------------程序结束！！！耗时：%s-----------"%during_time)

