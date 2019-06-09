import numpy as np
import pandas as pd

from setup import get_data
from binarized import get_binarized
from embeddings import get_embeddings

def get_counts_helper(df):
    fire   = df['pants on fire counts'].values.reshape(-1, 1)
    false  = df['false counts'].values.reshape(-1, 1)
    barely = df['barely true counts'].values.reshape(-1, 1)
    half   = df['half true counts'].values.reshape(-1, 1)
    mostly = df['mostly true counts'].values.reshape(-1, 1)

    combined = np.concatenate([fire, false, barely, half, mostly], axis=1)
    return combined

def get_counts():
    train, valid, test = get_data()
    a0 = get_counts_helper(train)
    a1 = get_counts_helper(valid)
    a2 = get_counts_helper(test)
    return [a0, a1, a2]

def get_meta(sep=False, most_common=500):
    state   = get_binarized('state info')
    subject = get_binarized('subject')
    speaker = get_binarized('speaker', most_common)

    job   = get_embeddings("speaker's job")
    venue = get_embeddings('venue')

    counts = get_counts()

    combined = [state, subject, speaker, job, venue, counts]

    comb_train = [i[0] for i in combined]
    comb_valid = [i[1] for i in combined]
    comb_test  = [i[2] for i in combined]

    if sep:
        return comb_train, comb_valid, comb_test

    meta_train = np.concatenate(comb_train, axis=1)
    meta_valid = np.concatenate(comb_valid, axis=1)
    meta_test  = np.concatenate(comb_test, axis=1)

    return meta_train, meta_valid, meta_test