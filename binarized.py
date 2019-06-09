"""
returns a multiclass binarized encoding of a column given its name

main outward facing functions:
get_binarized() takes in a column name
returns ndarray of shape (#examples, #classes)

recommended options for column name:
"state info" "subject" "speaker"
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from setup import get_train, get_valid, get_test

train = get_train(6)
valid = get_valid(6)
test = get_test(6)

#internal methods
def _get_classes(col):
    return list({i.strip().lower() for s in col if type(s) is not float for i in s.split(',') })

def _get_list(col):
    total = [[] for i in range(len(col))]
    for idx, s in enumerate(col):
        if type(s) is float: #NaN
            continue
        features = [i.strip().lower() for i in s.split(',')]
        total[idx] = features
    return total

def get_most_common(col, num_classes):
    job = col
    total = []
    for j in job:
        if type(j) is not float:
            total.append(j.strip())
    c = Counter(total)
    most = c.most_common(num_classes)
    return [m[0] for m in most]


def get_binarized(col, num_classes=None):
    if num_classes: #limit the amount of classes to a certain amount
        classes = get_most_common(train[col], num_classes)
    else:
        classes = _get_classes(train[col])
    processed_train = _get_list(train[col])
    enc = MultiLabelBinarizer(classes=classes)

    out_train = enc.fit_transform(processed_train)
    out_valid = enc.transform(_get_list(valid[col]))
    out_test  = enc.transform(_get_list(test[col]))

    return out_train, out_valid, out_test
