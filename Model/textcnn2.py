import pandas as pd
#import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding,Input,Conv1D
from keras.layers import Flatten, Dropout, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import concatenate
from keras.models import Sequential

from keras.models import Model
from keras.optimizers import *

## 先把原始的文本处理成2000维的向量，太长的截断，不够的补0
## 生成300维的嵌入
## CNN，3个256的卷积，池化以后，flatten，输入给softmax
## 输出分类的one hot编码


### 原始输入
# ~/Downloads/train_set.csv
# ~/workspace/sublime/daguan/train_sample.csv
path='E:/Heitao/DGB/数据集/'


doc_len = 2000
embedding_dim = 300


print('read data')
df_train = pd.read_csv(path + 'train_set.csv',engine='python',encoding='gbk')
df_test = pd.read_csv(path + 'train_set.csv',engine='python',encoding='gbk')
print (df_train.shape)
df_train.drop(df_train.columns[0],axis=1,inplace=True)
#df_train["word_seg"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
#df_test["word_seg"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)


word_seg = df_train['word_seg']
label = df_train['class'] - 1
X_train, X_test, y_train, y_test = train_test_split(word_seg, label, test_size=0.1, random_state=42)

print("encoding")
y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)


tokenizer = Tokenizer(split=' ')
tokenizer.fit_on_texts(word_seg)
vocab = tokenizer.word_index

# pad是填充，意思是在前面补零，处理完后长度均为200
# 输入转换

x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)


x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=doc_len)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=doc_len)




### RCNN模型
main_input = Input(shape=(doc_len,), dtype='float64')

embedder = Embedding(len(vocab) + 1, embedding_dim, input_length = doc_len)
embed = embedder(main_input)
# train a 1D convnet with global maxpoolinnb_wordsg
 
#left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）
model_left = Sequential()(embed)
#model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))

model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(35))
model_left.add(Flatten())
 
#right model 第二块神经网络，卷积窗口是4*50
 
model_right = Sequential()(embed)

model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(28))
model_right.add(Flatten())

model_3 = Sequential()(embed)

model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(30))
model_3.add(Flatten())
 
 
merged = concatenate([model_left,model_right,model_3], axis=-1) # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
model = Sequential()
model.add(merged) # add merge
model.add(Dense(128, activation='tanh')) # 全连接层
main_output=model.add(Dense(num_labels, activation='softmax')) # softmax，输出文本属于19种类别中每个类别的概率
model_end = Model(inputs = main_input, outputs = main_output)
model_end.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model_end.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

model_end.save('textcnn2.h5')

# 评价
score = model_end.evaluate(x_test_padded_seqs, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])	

## 特征转换
xx_test_word_ids = tokenizer.texts_to_sequences(df_test['word_seg'])
xx_test_padded_seqs = pad_sequences(xx_test_word_ids, maxlen=doc_len)

## 预测
pred_prob = model_end.predict(xx_test_padded_seqs)
pred = pred_prob.argmax(axis=1)


## 结果保存
df_test['class'] = pred.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('E:/Heitao/DGB/数据集/result/textcnn2.csv',index=False)
