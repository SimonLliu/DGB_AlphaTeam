import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding,Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model

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
df_train["word_seg"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
df_test["word_seg"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)


word_seg = df_train['word_seg']
label = df_train['class'] - 1
X_train, X_test, y_train, y_test = train_test_split(word_seg, label, test_size=0.1, random_state=42)

### label编码
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




### TextCNN模型
main_input = Input(shape=(doc_len,), dtype='float64')

embedder = Embedding(len(vocab) + 1, embedding_dim, input_length = doc_len)
embed = embedder(main_input)


cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)

cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)


flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)

main_output = Dense(num_labels, activation='softmax')(drop)

model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

model.save('textcnn.h5')

# 评价
score = model.evaluate(x_test_padded_seqs, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])	

## 特征转换
xx_test_word_ids = tokenizer.texts_to_sequences(df_test['word_seg'])
xx_test_padded_seqs = pad_sequences(xx_test_word_ids, maxlen=doc_len)

## 预测
pred_prob = model.predict(xx_test_padded_seqs)
pred = pred_prob.argmax(axis=1)


## 结果保存
df_test['class'] = pred.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('E:/Heitao/DGB/数据集/result/textcnn.csv',index=False)
