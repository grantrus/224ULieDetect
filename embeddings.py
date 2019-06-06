"""
Gets sentence-level embeddings by averaging word-by-word GloveEmbeddings.

main outward facing function:
get_embeddings() takes in one argument which is a column name
returns ndarray of shape is (num sentences, embedding size)

Can modify size of the glove embedding size. Options are {50, 100, 200, 300}. Default is 50

Best options are "speaker's job" and "venue"

TODO: experiment with tf-idf weighting of glove embeddings
"""

from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize as tokenize

#get data
from setup import get_train, get_valid, get_test
train = get_train(6)
valid = get_valid(6)
test = get_test(6)

GLOVE_SHAPE = 50 #can be changed

#import glove from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
GLOVE_HOME = os.path.join('..','data', 'glove.6B')
glove = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.' + str(GLOVE_SHAPE) + 'd.txt'))

def get_most_common(col):
    job = col
    total = []
    for j in job:
        if type(j) is not float:
            total.append(j.strip())
    c = Counter(total)
    most = c.most_common()
    return most

def get_glove(col):
    total = []
    for idx, s in enumerate(col):
        avg = np.zeros((1, GLOVE_SHAPE)) #initialize avg

        if type(s) is not float:
            tokenized = tokenize(s)
            for t in tokenized:
                t = t.lower()
                if t in glove: #ignore if not in glove dictionary
                    emb = glove.get(t).reshape(1, -1)
                    avg += emb
            if not len(tokenized):
                assert False, 'job at index {} is not valid'.format(idx)
            avg /= len(tokenized)

        total.append(avg) #if NaN (no job listed), embedding is a 0 array

    return np.concatenate(total, axis = 0)

def get_embeddings(col):
    return (get_glove(train[col]),
            get_glove(valid[col]),
            get_glove(test[col]))
