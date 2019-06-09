from meta import get_meta
import pandas as pd
from setup import get_data
import sys
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling1D, Dropout, Embedding, MaxPooling1D, Conv1D


num_labels = int(sys.argv[1])

#Get Embeddings from BERT
train_bert = np.load('data/X_bert_train_mean.npy')
test_bert = np.load('data/X_bert_test_mean.npy')
valid_bert = np.load('data/X_bert_valid_mean.npy')

#Get unaltered data which contains labels for each example.
train_data, valid_data, test_data = get_data(num_labels)
print(np.shape(train_bert))

meta_train, meta_valid, meta_test = get_meta(True, 100)

print(train_bert.shape)

train_all = np.concatenate((train_bert, meta_train[1]), axis=1)
test_all = np.concatenate((test_bert, meta_test[1]), axis=1)
valid_all = np.concatenate((valid_bert, meta_valid[1]), axis=1)

num_dims = train_all.shape[1]
print("Train Shape: ", train_all.shape)
print(test_all.shape)


model = Sequential()
seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(60, 768)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='models/modelMLP-{epoch:02d}.hdf5', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

model.fit(train_bert, train_data.label,
          epochs=300,
          batch_size=64,
          validation_data=(valid_bert, valid_data.label),
          callbacks=[checkpointer, earlystop])

print("Training Complete")
score = model.evaluate(train_bert, train_data.label, batch_size=64)
print("Train Score: ", score)
score = model.evaluate(valid_bert, valid_data.label, batch_size=64)
print("Valid Score: ", score)
score = model.evaluate(test_bert, test_data.label, batch_size=64)
print("Test Score: ", score)