"""
returns data, with meta-data pre-processed

usage:
from setup import get_train
train = get_train(3) #returns train with 3 labels: 0, 1, 2

details:
Has 3 outward facing functions that all take 2, 3, 6 as input (amount of labels)
- get_train()
- get_valid()
- get_test()
"""

import pandas as pd

def get_data(num_labels=6):
    train = get_train(num_labels)
    valid = get_valid(num_labels)
    test  = get_test(num_labels)
    return train, valid, test

def preprocess(df, num_labels):
    set_header(df)
    set_labels(df, num_labels)
    set_party(df)
    append_lying_ratio(df)
    append_ratio_significance(df)

def get_train(num_labels):
    train = pd.read_csv('data/train.csv', header=None)
    preprocess(train, num_labels)
    return train

def get_valid(num_labels):
    valid = pd.read_csv('data/valid.csv', header=None)
    preprocess(valid, num_labels)
    return valid

def get_test(num_labels):
    test = pd.read_csv('data/test.csv', header=None)
    preprocess(test, num_labels)
    return test

def set_labels(df, num_labels):
    if num_labels == 2: set_labels2(df)
    elif num_labels == 3: set_labels3(df)
    elif num_labels == 6: set_labels6(df)
    else:
        assert False, "{} is not a valid amount of labels. Only 2, 3, 6 are supported".format(num_labels)

def set_labels2(df):
    """True-ish is 1, false-ish is 0"""
    new_labels = []
    for l in df.label:
        if l == 'pants-fire' or l == 'FALSE' or l == "barely-true" : new_labels.append(0)
        elif l == "half-true" or l == "mostly-true" or l == "TRUE": new_labels.append(1)
        else:
            assert False, "{} is not a normal label".format(l)
    df.label = new_labels

def set_labels3(df):
    """True-ish is 2, kinda true is 1, false-ish is 0"""
    new_labels = []
    for l in df.label:
        if l == 'pants-fire' or l == 'FALSE': new_labels.append(0)
        elif l == "barely-true" or l == "half-true": new_labels.append(1)
        elif l == "mostly-true" or l == "TRUE": new_labels.append(2)
        else:
            assert False, "{} is not a normal label".format(l)
    df.label = new_labels

def set_labels6(df):
    """ranges from 5 which is True, to 0 which is pants-fire"""
    new_labels = []
    for l in df.label:
        if l == 'pants-fire': new_labels.append(0)
        elif l == 'FALSE': new_labels.append(1)
        elif l == "barely-true": new_labels.append(2)
        elif l == "half-true": new_labels.append(3)
        elif l == "mostly-true": new_labels.append(4)
        elif l == "TRUE": new_labels.append(5)
        else: assert False, "{} is not a normal label".format(l)
    df.label = new_labels

def set_party(df):
    """Conservative Spectrum"""
    party = []
    for p in df.party:
        if p == 'republican':
            party.append(1)
        elif p == "democrat" :
            party.append(0)
        else:
            party.append(.5)
    df.party = party

def append_lying_ratio(df):
    """
    creates a weighted average of truth history from 0 to 1 per statement
    appends it to the train df as 'lying ratio'
    the value ranges from 0 to 1
    where 0 is they always tell truth and 1 is they always lie
    if they don't have any history set as a random number (will be offset by 'truth history' which will be set to 0)

    TODO: subtract current label since included in counts (recommended in LIAR paper)
    """

    ratio = []

    fire = df['pants on fire counts']
    false = df['false counts']
    barely = df['barely true counts']
    half = df['half true counts']
    mostly = df['mostly true counts']

    FIRE_W = 1
    FALSE_W = .8
    BARELY_W = .6
    HALF_W = .4
    MOSTLY_W = .2

    RANDOM = .5 #if no history set to this value

    for i in range(len(df)):
        avg = 0
        avg += fire[i] * FIRE_W
        avg += false[i] * FALSE_W
        avg += barely[i] * BARELY_W
        avg += half[i] * HALF_W
        avg += mostly[i] * MOSTLY_W

        total = fire[i] + false[i] + barely[i] + half[i] + mostly[i]

        if total == 0:
            ratio.append(RANDOM)
        else:
            avg /= total
            ratio.append(avg)

    df['lying ratio'] = ratio

def append_ratio_significance(df):
    """
    sets significance as a value between 0 and 1
    appends it to df as 'ratio significance'
    """
    sig =  []

    fire = df['pants on fire counts']
    false = df['false counts']
    barely = df['barely true counts']
    half = df['half true counts']
    mostly = df['mostly true counts']

    CUTOFF = 100

    for i in range(len(df)):
        total = fire[i] + false[i] + barely[i] + half[i] + mostly[i]
        if total > CUTOFF:
            sig.append(1)
        else:
            sig.append(total / CUTOFF)

    df['ratio significance'] = sig

def set_header(df):
    HEADER = ["ID", "label", "statement", "subject", "speaker", "speaker's job", "state info", "party", "barely true counts", "false counts", "half true counts", "mostly true counts", "pants on fire counts", "venue"]
    df.columns = HEADER

if __name__ == "__main__":
    print('shape of training data is: ' + str(get_train(6).shape))