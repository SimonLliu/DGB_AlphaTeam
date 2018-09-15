# -*- coding: utf-8 -*-


## 对特征进行嵌入式选择
import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

t_start = time.time()

print('读取特征...')
"""读取特征"""
features_path = 'D:/p_project/try_again/feature/TfidfVectorizer_article/data_tfidf_article.pkl'

#tfidf特征的路径
with open(features_path, 'rb') as fp:
    x_train, y_train, x_test = pickle.load(fp)


print('进行特征选择...')
"""进行特征选择"""

LR = LogisticRegression(C=10, dual=False).fit(x_train, y_train)
slt = SelectFromModel(LR, prefit=True)
x_train_s = slt.transform(x_train)
x_test_s = slt.transform(x_test)


print('保存选择后的特征至本地...')
"""保存选择后的特征至本地"""
num_features = x_train_s.shape[1]
data_path = 'D:/p_project/try_again/feature/TfidfVectorizer_article/LR_selectfeature_article.pkl'
with open(data_path, 'wb') as data_f:
    pickle.dump((x_train_s, y_train, x_test_s), data_f)

t_end = time.time()
print("特征选择完成，选择{}个特征，共耗时{}min".format(num_features, (t_end-t_start)/60))
