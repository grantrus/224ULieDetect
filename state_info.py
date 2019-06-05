"""
returns a one hot encoding for the state's of train, valid, and test

outward facing functions:
get_state_train()
get_state_valid()
get_state_test()
get_classes()
"""

from sklearn.preprocessing import MultiLabelBinarizer

from setup import get_train, get_valid, get_test
from subject import _get_list, _get_classes

train, valid, test = get_train(6), get_valid(6), get_test(6)

#initialize by fitting the encoder to train
classes = _get_classes(train['state info'])
enc = MultiLabelBinarizer(classes=classes)
processed_train = _get_list(train['state info'])
enc.fit(processed_train)

def get_classes():
    return classes

def get_state_train():
    return enc.transform(processed_train)

def get_state_valid():
    return enc.transform(_get_list(valid['state info']))

def get_state_test():
    return enc.transform(_get_list(test['state info']))


