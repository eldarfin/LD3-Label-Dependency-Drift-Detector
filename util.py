import numpy as np 
import pandas as pd 
from ast import literal_eval
from skmultilearn.dataset import load_dataset
import matplotlib.pyplot as plt
import operator


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
    Y = np.array(Y)
    num_labels = Y.shape[1]
    freqs = np.matrix(np.zeros(shape=(num_labels, num_labels), dtype=np.int32))
    for label in Y:
        l = np.matrix(label)
        a = np.matmul(np.transpose(l), l, dtype=np.int32)
        freqs = np.add(freqs, a)
    if not self_loops:    
        freqs = np.asarray(freqs)
        np.fill_diagonal(freqs, 0)
    #freqs = np.matrix(freqs)
    return np.asarray(freqs)

def recip_rank(mat):
    num_labels = len(mat)
    ranks = np.flip(np.argsort(mat), axis=1)
    ranks = np.delete(ranks, [np.argwhere(i == ranks[i]).flatten()[0]+i*num_labels for i in range(num_labels)]).reshape(num_labels,num_labels-1)
    return np.argsort([1/np.sum(1/(np.argwhere(ranks==i)[:, 1] + 1)) for i in range(num_labels)])
    '''ranks = []
    for i, row in enumerate(mat):
        row_ = np.array(row)[0]
        ranking = np.flip(np.argsort(row_))
        ranking = np.delete(ranking, np.argwhere(ranking == i))
        ranks.append(ranking)

    f_ranks = np.zeros(num_labels)
    for rank in ranks:
        for i, r in enumerate(rank):
            f_ranks[r] += 1/(i+1)
    f_ranks = 1 / f_ranks
    return np.argsort(f_ranks)'''

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

