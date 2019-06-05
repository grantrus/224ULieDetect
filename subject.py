"""
returns a multiclass one hot encoding for the subject's of train, valid, and test

outward facing functions:
get_subject_train()
get_subject_valid()
get_subject_test()
get_classes()
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from setup import get_train, get_valid, get_test

train = get_train(6)
valid = get_valid(6)
test = get_test(6)

#internal methods
def _get_classes(col):
    return list({i.strip().lower() for s in col for i in s.split(',')})

def _get_list(col):
    total = [[] for i in range(len(col))]
    for idx, s in enumerate(col):
        if type(s) is float: #NaN
            continue
        features = [i.strip().lower() for i in s.split(',')]
        total[idx] = features
    return total

#initialize by fitting the encoder to train)
classes = list(_get_classes(train.subject))
enc = MultiLabelBinarizer(classes=classes)
processed_train = _get_list(train.subject)
enc.fit(processed_train)

def get_classes():
    return classes

def get_subject_train():
    return enc.transform(processed_train)

def get_subject_valid():
    return enc.transform(_get_list(valid.subject))

def get_subject_test():
    return enc.transform(_get_list(test.subject))


