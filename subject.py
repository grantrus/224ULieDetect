"""
returns a multiclass one hot encoding for the subject's of train, valid, and test

outward facing functions:
get_subject_train()
get_subject_valid()
get_subject_test()
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from setup import get_train, get_valid, get_test

train = get_train(6)
valid = get_valid(6)
test = get_test(6)

#important methods
def get_classes(col):
    return list({i.strip().lower() for s in col for i in s.split(',')})

def get_list(col):
    total = [[] for i in range(len(col))]
    for idx, s in enumerate(col):
        if type(s) is float: #NaN
            continue
        features = [i.strip().lower() for i in s.split(',')]
        total[idx] = features
    return total

#initialize by fitting the encoder to train)
classes = list(get_classes(train.subject))
enc = MultiLabelBinarizer(classes=CLASSES)
processed_train = get_list(train.subject)
enc.fit(processed_train)

def get_subject_train():
    return enc.transform(processed_train)

def get_subject_valid():
    return enc.transform(get_list(valid.subject))

def get_subject_test():
    return enc.transform(get_list(test.subject))


