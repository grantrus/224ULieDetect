from setup import get_train, get_test, get_valid
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Embedding, SpatialDropout1D, Input, Bidirectional

train_bert = np.load('data/X_bert_train_mean.npy')
test_bert = np.load('data/X_bert_test_mean.npy')
print(np.shape(train_bert))
num_labels = int(sys.argv[1])

train_data = get_train(num_labels)
test_data = get_test(num_labels)
print(train_bert.shape)
#print(train_bert[0])

MAXLEN = 876
print(train_bert[0])

abs_min = float('inf')
for i in range(train_bert.shape[0]):
    cur_min = min(train_bert[i])
    if cur_min < abs_min:
        abs_min = cur_min
print(abs_min)

abs_min2 = float('inf')
for i in range(test_bert.shape[0]):
    cur_min2 = min(train_bert[i])
    if cur_min2 < abs_min2:
        abs_min2 = cur_min2
print(abs_min2)
abs_min3 = min(abs_min, abs_min2)

for i in range(test_bert.shape[0]):
    test_bert[i] += abs(abs_min3)

for i in range(train_bert.shape[0]):
    train_bert[i] += abs(abs_min3)

print(train_bert[0])
np.save('data/X_bert_train_pos.npy', train_bert)
np.save('data/X_bert_test_pos.npy', train_bert)

model = Sequential()
model.add(Embedding(30523, 100, input_length=768))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))

#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

checkpointer = ModelCheckpoint(filepath='models/model2-{epoch:02d}.hdf5', verbose=1)
history = model.fit(train_bert, train_data.label, epochs=5, batch_size=8, validation_data=(test_bert, test_data.label), callbacks=[checkpointer])

'''
model = Sequential()
model.add(Embedding(30523, 50, input_length=768))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Dense(64, name='FC1', activation='relu'))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

'''