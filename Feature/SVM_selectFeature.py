# -*- coding: utf-8 -*-


## 对特征进行嵌入式选择
import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

t_start = time.time()

print('读取特征...')
"""读取特征"""
features_path = './feature/TfidfVectorizer_article/data_tfidf_article.pkl'

#tfidf特征的路径
fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(fp)
fp.close()


print('进行特征选择')
"""进行特征选择"""
alo_name = 'LSVC_l2'
lsvc = LinearSVC(C=0.5, dual=False).fit(x_train, y_train)
slt = SelectFromModel(lsvc, prefit=True)
x_train_s = slt.transform(x_train)
x_test_s = slt.transform(x_test)


print('保存选择后的特征至本地...')
"""保存选择后的特征至本地"""
num_features = x_train_s.shape[1]
data_path = 'data_selectfeature_article.pkl'
data_f = open(data_path, 'wb') 
pickle.dump((x_train_s, y_train, x_test_s), data_f)
data_f.close()

t_end = time.time()
print("特征选择完成，选择{}个特征，共耗时{}min".format(num_features, (t_end-t_start)/60))



