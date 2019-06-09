from meta import get_meta
import pandas as pd
from setup import get_data
import sys
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Embedding, SpatialDropout1D, Input, Bidirectional


num_labels = int(sys.argv[1])

#Get Embeddings from BERT
train_bert = np.load('data/X_bert_train_mean.npy')
test_bert = np.load('data/X_bert_test_mean.npy')
valid_bert = np.load('data/X_bert_valid_mean.npy')

#Get unaltered data which contains labels for each example.
train_data, valid_data, test_data = get_data(num_labels)
print(np.shape(train_bert))

meta_train, meta_valid, meta_test = get_meta(False, 100)

print(train_bert.shape)

train_all = np.concatenate((train_bert, meta_train), axis=1)
test_all = np.concatenate((test_bert, meta_test), axis=1)
valid_all = np.concatenate((valid_bert, meta_valid), axis=1)

num_dims = train_all.shape[1]
print("Train Shape: ", train_all.shape)
print(test_all.shape)

#assert not np.any(np.isnan(train_all))
model = Sequential()

model.add(Dense(64, activation='relu', input_dim=num_dims))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(num_labels, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='models/modelMLP-{epoch:02d}.hdf5', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

model.fit(train_all, train_data.label,
          epochs=3000,
          batch_size=64,
          validation_data=(valid_all, valid_data.label),
          callbacks=[checkpointer, earlystop])

print("Training Complete")
score = model.evaluate(train_all, train_data.label, batch_size=64)
print("Train Score: ", score)
score = model.evaluate(valid_all, valid_data.label, batch_size=64)
print("Valid Score: ", score)
score = model.evaluate(test_all, test_data.label, batch_size=64)
print("Test Score: ", score)