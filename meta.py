import numpy as np
import pandas as pd

from setup import get_data
from binarized import get_binarized
from embeddings import get_embeddings

def get_numerical(col):
    train, valid, test = get_data()

    a0 = np.array(train[col]).reshape(-1, 1)
    a1 = np.array(valid[col]).reshape(-1, 1)
    a2 = np.array( test[col]).reshape(-1, 1)
    return [a0, a1, a2]

def get_meta(most_common=500):
    state   = get_binarized('state info')
    subject = get_binarized('subject')
    speaker = get_binarized('speaker', most_common)

    job   = get_embeddings("speaker's job")
    venue = get_embeddings('venue')

    ratio = get_numerical('lying ratio')
    sig   = get_numerical('ratio significance')

    combined = [state, subject, speaker, job, venue, ratio, sig]

    meta_train = np.concatenate([i[0] for i in combined], axis=1)
    meta_valid = np.concatenate([i[1] for i in combined], axis=1)
    meta_test  = np.concatenate([i[2] for i in combined], axis=1)

    return meta_train, meta_valid, meta_test