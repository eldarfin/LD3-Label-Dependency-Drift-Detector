import numpy as np 
import pandas as pd 
from ast import literal_eval
from scipy.spatial import distance
from skmultilearn.dataset import load_dataset
import matplotlib.pyplot as plt
import operator
import scipy
import time


def WS(r1,r2):
        sum_ = 0
        ranks_x = np.argsort(r1)
        ranks_y = np.argsort(r2)
        for i in range(len(r1)):
            sum_ += (1/(2**(ranks_x[i]))) * ((np.abs(ranks_x[i] - ranks_y[i]))/(np.max([np.abs(1-ranks_x[i]), np.abs(len(r1) - ranks_x[i])])))
        return 1 - sum_

def generate_csv(dataset, save_dir='./', split='train', self_loops=False):
    X_train, y_train, _, _ = load_dataset(dataset, split)
    y_train = y_train.toarray()
    X_train = X_train.toarray()


    features = ["{}".format(i) for i in range(X_train.shape[1])]
    labels = ["{}".format(i+X_train.shape[1]) for i in range(y_train.shape[1])]
    
    X = pd.DataFrame(data=X_train, columns=features)
    X.index.name = 'index'
    Y = pd.DataFrame(data=y_train, columns=labels)
    X.index.name = 'index'
    
    df = pd.merge(X, Y, right_index=True, left_index=True)
    df.to_csv('./{}/{}.csv'.format(save_dir, dataset), sep='\t', encoding='utf-8')

def load_data(filename, in_dir='./', dtype='dataframe'):
    '''rows = None
    with open('{}/scene_ranking.txt'.format(in_dir), 'r') as f:
        rows = f.readlines()
    A = [literal_eval(row.strip()) for row in rows]
    A = [[r[1] for r in row] for row in A]
    A = np.matrix(A)
    return A'''
    df = pd.read_csv('{}/{}'.format(in_dir, filename), sep='\t', index_col='index')
    
    if dtype == 'dataframe':
        return df
    else:
        return np.matrix(df.values)
def to_numpy_matrix(Y, self_loops=False):
    '''num_labels = Y.shape[1]
    freqs = np.matrix(np.zeros(shape=(num_labels, num_labels), dtype=np.int32))
    for label in Y:
        l = np.matrix(label)
        a = np.matmul(np.transpose(l), l, dtype=np.int32)
        freqs = np.add(freqs, a)'''
    t = np.transpose(Y)
    freqs = t@Y
    if not self_loops:    
        np.fill_diagonal(freqs, 0)
    #freqs = np.matrix(freqs)
    return freqs

def w(r):
    if r < 2:
        return r
    else:
        return 0

def subset_accuracy(y_true, y_pred):
    N, l = y_true.shape
    accuracy = 0
    for i in range(N):
        and_sum = 0
        or_sum = 0
        for j in range(l):
            if y_true[i][j] == y_pred[i][j]:
                and_sum += 1
            if y_true[i][j] == 1 or y_pred[i][j] == 1:
                or_sum += 1
        if or_sum == 0:
            accuracy += 0
        else:
            accuracy += and_sum / or_sum
    return accuracy / N

def recip_rank(mat):
    num_labels = len(mat)
    ranks = np.flip(np.argsort(mat), axis=1)
    ranks = np.delete(ranks, [np.argwhere(i == ranks[i]).flatten()[0]+i*num_labels for i in range(num_labels)]).reshape(num_labels,num_labels-1)
    return np.argsort([1/np.sum(1/(np.argwhere(ranks==i)[:, 1] + 1)) for i in range(num_labels)])

def add_to_pretrain(X, y, sample, label, size):
    if len(X) < size:
        X.append(sample)
        y.append(label)
    else:
        X.pop(0)
        X.append(sample)
        y.pop(0)
        y.append(label)
    return X, y 


def add_left_shift_drift(y, start_point, shift_amount=2):
    new_y = []
    for i, label in enumerate(y):
        label = label.tolist()
        if i >= start_point:
            for i in range(shift_amount):
                elem = label.pop(0)
                label.append(elem)
        new_y.append(np.array(label))
    return np.array(new_y)

def add_label_count_drift(y, start_point, count=3):
    new_y = []
    for i, label in enumerate(y):
        label = label.tolist()
        if i >= start_point and i % 3 == 0:
            for j in range(count):
                label[j] = 1
        new_y.append(label)
    new_y.reverse()
    return np.array(new_y)

