from meta import get_meta
from setup import get_data
import sys
import numpy as np


num_labels = int(sys.argv[1])

#Get Embeddings from BERT
train_bert = np.load('data/X_bert_train_mean.npy')
test_bert = np.load('data/X_bert_test_mean.npy')
valid_bert = np.load('data/X_bert_valid_mean.npy')

#Get unaltered data which contains labels for each example.
train_data, valid_data, test_data = get_data(num_labels)
#print(np.shape(train_bert))

meta_train, meta_valid, meta_test = get_meta(True, 100)

train = np.array(meta_train)
print(train)

print(train_bert.shape)

train_all = np.concatenate((train_bert, meta_train[1]), axis=1)
test_all = np.concatenate((test_bert, meta_test[1]), axis=1)
valid_all = np.concatenate((valid_bert, meta_valid[1]), axis=1)

print(train_all.shape)
print(test_all.shape)
print(valid_all.shape)
