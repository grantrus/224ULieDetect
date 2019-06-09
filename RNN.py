from setup import get_train, get_test, get_valid
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Embedding, SpatialDropout1D, InputLayer

num_labels = int(sys.argv[1])
train_data = get_train(num_labels)
test_data = get_test(num_labels)
#val_data = get_valid(num_labels)

both = [train_data, test_data]
combined = pd.concat(both)

train_s = train_data.statement
test_s = test_data.statement
#val_s = val_data.statement

#Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(combined.statement)

X_train = vectorizer.transform(train_s)
X_test = vectorizer.transform(test_s)
#X_val = vectorizer.transform(val_s)

print(X_train.shape)
model = Sequential()
model.add(Embedding(1000, 50, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))

#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, train_data.label, epochs=5, batch_size=8,validation_data=(X_test, test_data.label))