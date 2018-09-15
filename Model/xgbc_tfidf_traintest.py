
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import time

import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


'''
print('读取数据')
df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')

df_train.drop(columns=['article','id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

# 提取tfidf文本特征
# vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])

# 训练集
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class']

# 测试集
x_test = vectorizer.transform(df_test['word_seg'])
'''

#

'''
from scipy import sparse


sparse.save_npz("data/tfidf/x_train.npz", x_train)
y_train.to_csv('data/tfidf/y_train.csv')
sparse.save_npz("data/tfidf/x_test.npz", x_test)


from scipy import sparse
x_train = sparse.load_npz("data/tfidf/x_train.npz")
y_train = pd.read_csv('data/tfidf/y_train.csv')
x_test = sparse.load_npz("data/tfidf/x_test.npz")

# XGBoost训练过程
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

'''

## 抽样
print('读取特征:')
#读取Model
import pickle

with open('./feature/tfidf/x_train2.pickle', 'rb') as f:
    x_train = pickle.load(f)

df_train = pd.read_csv('~/Downloads/train_set.csv')
df_test = pd.read_csv('~/Downloads/test_set.csv')
y_train = df_train['class'] - 1
#y_train.to_csv('./feature/y_train.csv')

with open('./feature/tfidf/x_test2.pickle', 'rb') as f3:
    x_test = pickle.load(f3)

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=0)

print('开始训练:')
start = time.time() 

# max_depth=15, n_estimators=30, learning_rate=0.05
# 这里还可以尝试其他多种模型，利用fit()函数和预测predict(),这里使用XGboost
gbm = xgb.XGBClassifier().fit(X_train, y_train)
print('训练集上的误差：{}'.format(gbm.score(X_train,y_train)))


pred_test = gbm.predict(X_test)
accuracy = accuracy_score(y_test, pred_test)
print("测试集上的准确率:{}".format(accuracy))

print("测试集报告")
print(classification_report(y_test, pred_test))

end = time.time()
print('time',end-start)



y_test = gbm.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
#df_result.to_csv('./result/result-xgbc.csv', index=False)

# 显示重要特征
#plot_importance(gbm)
#plt.show()
