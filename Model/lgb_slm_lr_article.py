# -*- coding: utf-8 -*-
"""
@brief : lgb算法
@author: Jian
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
import lightgbm as lgb
from sklearn.externals import joblib

t_start = time.time()

"""=====================================================================================================================
0 自定义验证集的评价函数
"""
def f1_score_vali(preds, data_vali):
    
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(19, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True

"""=====================================================================================================================
1 读取数据,并转换到lgb的标准数据格式
"""
#features_path = 'D:/DGB/feature_merge/LR_selectfeature_article.pkl'
data_fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

"""划分训练集和验证集，验证集比例为test_size"""
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
d_train = lgb.Dataset(data=x_train, label=y_train)
d_vali = lgb.Dataset(data=x_vali, label=y_vali)

"""=====================================================================================================================
2 训练lgb分类器
"""
params = {
        'boosting': 'gbdt',
        'application': 'multiclassova',
        'num_class': 19,
        'learning_rate': 0.1,
        'num_leaves':31,
        'max_depth':-1,
        'lambda_l1': 0,
        'lambda_l2': 0.5,
        'bagging_fraction' :1.0,
        'feature_fraction': 1.0
        }

bst = lgb.train(params, d_train, num_boost_round=800, valid_sets=d_vali,feval=f1_score_vali, early_stopping_rounds=None,
                verbose_eval=True)
#path='D:/DGB/feature_merge/'
joblib.dump(bst, path+"lgb_(lsm_lr_article)).m")
 
"""=====================================================================================================================
3 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地
"""
y_proba = bst.predict(x_test)
y_test = np.argmax(y_proba, axis=1) + 1

df_result = pd.DataFrame(data={'id':range(102277), 'class': y_test.tolist()})
df_proba = pd.DataFrame(data={'id':range(102277), 'proba': y_proba.tolist()})
#result_path = '../results/' + features_path.split('/')[-1] + '_lgb' + '.csv'
#df_result.to_csv(result_path, index=False

df_result.to_csv(path + 'lgb_(lsm_lr_article))_result.csv',index=False)
df_proba.to_csv(path + 'lgb_(lsm_lr_article))_proba.csv',index=False)
t_end = time.time()
print("训练结束，耗时:{}min".format((t_end - t_start) / 60))


