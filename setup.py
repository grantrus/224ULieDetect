"""
Has 3 outward facing functions that return the respective data, with meta-data pre-processed
- get_train()
- get_valid()
- get_test()
"""

import pandas as pd

def set_labels(df):
    """True-ish is 2, kinda true is 1, false-ish is 0"""
    new_labels = []
    for l in df.label:
        if l == 'pants-fire' or l == 'FALSE':
            new_labels.append(0)
        elif l == "barely-true" or l == "half-true":
            new_labels.append(1)
        elif l == "mostly-true" or l == "TRUE":
            new_labels.append(2)
        else:
            assert False, "{} is not a normal label".format(l)
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

    for i in range(len(train)):
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

    for i in range(len(train)):
        total = fire[i] + false[i] + barely[i] + half[i] + mostly[i]
        if total > CUTOFF:
            sig.append(1)
        else:
            sig.append(total / CUTOFF)

    df['ratio significance'] = sig

def set_header(df):
    HEADER = ["ID", "label", "statement", "subject", "speaker", "speaker's job", "state info", "party", "barely true counts", "false counts", "half true counts", "mostly true counts", "pants on fire counts", "venue"]
    df.columns = HEADER

def preprocess(df):
    set_labels(df)
    set_party(df)
    append_lying_ratio(df)
    append_ratio_significance(df)

def get_train():
    train = pd.read_csv('data/train.csv', header=None)
    preprocess(train)
    return train

def get_valid():
    valid = pd.read_csv('data/valid.csv', header=None)
    preprocess(valid)
    return valid

def get_test():
    test = pd.read_csv('data/test.csv', header=None)
    preprocess(test)
    return test
