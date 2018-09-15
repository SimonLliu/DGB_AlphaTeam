# -*- coding: utf-8 -*-
"""
@简介：将data_ensemble特征转换为稀疏矩阵，并将其合并到tfidf
@author: Jian
"""
import pickle
from scipy import sparse
from scipy.sparse import hstack

"""读取ensemble特征"""
f_1 = open('./data_hash_article.pkl', 'rb')
x_train_1, y_train, x_test_1 = pickle.load(f_1)
f_1.close()

f_2 = open('./data_doc2vec_article.pkl', 'rb')
x_train_2, _, x_test_2 = pickle.load(f_2)
f_2.close()

f_3 = open('./data_doc2vec_word.pkl', 'rb')
x_train_3, _, x_test_3 = pickle.load(f_3)
f_3.close()

f_4 = open('./data_hash_word.pkl', 'rb')
x_train_4, _, x_test_4 = pickle.load(f_4)
f_4.close()

"""将numpy 数组 转换为 csr稀疏矩阵"""
x_train_1 = sparse.csr_matrix(x_train_1)
x_test_1 = sparse.csc_matrix(x_test_1)

x_train_2 = sparse.csr_matrix(x_train_2)
x_test_2 = sparse.csc_matrix(x_test_2)

x_train_3 = sparse.csr_matrix(x_train_3)
x_test_3 = sparse.csc_matrix(x_test_3)

x_train_4 = sparse.csr_matrix(x_train_4)
x_test_4 = sparse.csc_matrix(x_test_4)
"""读取tfidf特征"""
#f_tfidf = open('./data_tf_article.pkl', 'rb')
#x_train_3, _, x_test_3= pickle.load(f_tfidf)
#f_tfidf.close()

"""对两个稀疏矩阵进行合并"""
x_train_5 = hstack([x_train_1, x_train_2])
x_test_5 = hstack([x_test_1, x_test_2])

x_train_6 = hstack([x_train_5, x_train_3])
x_test_6 = hstack([x_test_5, x_test_3])

x_train_7 = hstack([x_train_6, x_train_4])
x_test_7 = hstack([x_test_6, x_test_4])
"""将合并后的稀疏特征保存至本地"""
data = (x_train_7, y_train, x_test_7)
f = open('./data_ensemble_a(doc+hash)_w(doc+hash).pkl', 'wb')
pickle.dump(data, f)
f.close()




