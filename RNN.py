from setup import get_train, get_test, get_valid
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed

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

model = Sequential()
model.add(64, input_length=X_train.shape[1])
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(128)))
model.add(TimeDistributed(Dense(64)))
model.add(Activation('softmax'))

#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.fit(x=X_train, y=train_data.label, batch_size=8, validation_data=(X_test, test_data.label))