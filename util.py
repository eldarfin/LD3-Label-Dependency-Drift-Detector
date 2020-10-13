import numpy as np 
import pandas as pd 
from ast import literal_eval
from skmultilearn.dataset import load_dataset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx


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
    freqs = np.matrix(freqs)
    return freqs

def recip_rank(mat):
    ranks = []
    for i, row in enumerate(mat):
        row_ = np.array(row)[0]
        ranking = np.flip(np.argsort(row_))
        ranking = np.delete(ranking, np.argwhere(ranking == i))
        ranks.append(ranking)

    f_ranks = np.zeros(mat.shape[0])
    for rank in ranks:
        for i, r in enumerate(rank):
            f_ranks[r] += 1/(i+1)
    f_ranks = 1 / f_ranks
    return np.argsort(f_ranks)

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

def generate_graph(A, in_dir='./', filename=None, from_file=False, self_loops=True):
    if from_file:
        if filename is None:
            raise Exception('An input filename should be provided')
        A = load_data(filename, in_dir)
    
    if type(A) is not np.matrix:
        A = to_numpy_matrix(A, self_loops=self_loops)

    G = nx.from_numpy_matrix(A)
    return G 

def draw_graph(G, save_dir=None, filename=None, pos=None, labels=None, with_weights=False, with_labels=False):
    if pos is None:
        pos = nx.spring_layout(G, weight='weight')
    if labels is None:
        labels = {i:str(i) for i in range(len(G.nodes()))}
    
    nx.draw_networkx(G, pos=pos, labels=labels, with_labels=with_labels, alpha=0.5) 
    if with_weights:
        edge_labels=nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    
    if save_dir is None:
        plt.show()
    else:
        if filename is None:
            raise Exception('An output filename should be provided')
        else:
            plt.savefig('{}/{}'.format(save_dir, filename))
    
    plt.close()

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


def get_cluster_count(G):
    A = nx.to_numpy_matrix(G, dtype=np.int32)
    t = 0
    for i in range(A.shape[0]*A.shape[1]):
        if A.item(i) != 0:
            t += 1
    return int(np.floor(A.shape[0] * A.shape[1] / t))

def is_independent(G):
    num_labels = len(G.nodes())
    isolates = get_isolates(G)
    num_isolates = len(isolates)
    print('Independent:', num_labels == num_isolates) 
    flag = True
    for n in isolates:
        if n and flag:
            print('Independent labels:')
            flag = False
        print(n)
    return num_labels == num_isolates
    
def get_isolates(G):
    isolates = [i for i in nx.isolates(G)]
    return isolates
