import numpy as np
import operator
import util
import pandas as pd


class LD3():
    def __init__(self, k=2, window_size=500, detection_window_size=1, label_count=10000, big_dataset=False):
        self.history = {}
        self.prev = []
        self.detection_window_size = detection_window_size
        self.k = k
        self.window_size = window_size
        self.labels = []
        self.warmup = True
        #self.label_count = label_count
        self.big_dataset = big_dataset
        self.decrease_rate = 0
        self.increase_rate = 1

    def add_element(self, y):
        if len(self.labels) < self.window_size:
                self.labels.extend(y)
        '''if self.big_dataset:
            if len(self.labels) < self.label_count:
                self.labels.extend(y)
            else:
                self.labels.pop(0)
                self.labels.extend(y)
        else:
            self.labels.extend(y)'''

    def recip_rank(self, mat):
        num_labels = len(mat)
        ranks = np.flip(np.argsort(mat), axis=1)
        ranks = np.delete(ranks, [np.argwhere(i == ranks[i]).flatten()[0]+i*num_labels for i in range(num_labels)]).reshape(num_labels,num_labels-1)
        return np.argsort([1/np.sum(1/(np.argwhere(ranks==i)[:, 1] + 1)) for i in range(num_labels)])
        '''ranks = []
        for i, row in enumerate(mat):
            ranking = np.flip(np.argsort(row))
            ranking = np.delete(ranking, np.argwhere(ranking == i))
            ranks.append(ranking)

        f_ranks = np.zeros(mat.shape[0])
        for rank in ranks:
            for i, r in enumerate(rank):
                f_ranks[r] += 1/(i+1)
        f_ranks = 1 / f_ranks
        return np.argsort(f_ranks)'''

    def add_to_history(self, item):
        if item not in self.history:
            self.history[item] = 1
        else:
            if self.history[item] < 0:
                self.history[item] = 0
            #self.history[item] += 1
            current_hist = np.array(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))[:self.k][:,0]
            if item not in current_hist and item == self.prev:
                self.history[item] += 2 ** self.increase_rate
            else:
                self.history[item] += 1
        '''if len(self.history) < self.detection_window_size:
            self.history.append(item)
        else:
            self.history.pop(0)
            self.history.append(item)'''

    def detected_change(self):
        if len(self.labels) % self.window_size == 0 and len(self.labels) != 0:
            print()
            if len(self.prev) < 1:
                r = self.recip_rank(util.to_numpy_matrix(self.labels))[:self.k]
                for item in r:
                    self.add_to_history(item)
                print(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                #self.prev = r
                self.prev = np.array(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True))[:self.k])[:,0]
                self.labels = []   
            else:
                r = self.recip_rank(util.to_numpy_matrix(self.labels))[:self.k]
                for item in r:
                    self.add_to_history(item)
                
                #diff = np.setdiff1d(r, self.prev)
                current_hist = np.array(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                if current_hist[0][1] <= current_hist[1][1] and self.warmup:
                    print('Warmup')
                    print(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                    print(r)
                    self.labels = []
                    self.prev = current_hist[:self.k][:, 0]
                    return False
                elif current_hist[0][1] > current_hist[1][1] and self.warmup:
                    print('Warmup ended')
                    print(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                    print(r)
                    self.labels = []
                    self.prev = current_hist[:self.k][:, 0]
                    self.warmup = False
                    return False
                    

                if r[0] != self.prev[0]:
                    self.history[self.prev[0]] -= 2 ** self.decrease_rate
                    current_hist = np.array(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                    self.decrease_rate += 1
                else:
                    self.decrease_rate = 0
                
                print(list(sorted(self.history.items(), key=operator.itemgetter(1),reverse=True)))
                print(r)

                diff = np.setdiff1d(current_hist[:self.k][:, 0], self.prev)
                self.labels = []
                #self.prev = r
                self.prev = current_hist[:self.k][:, 0]
                if len(diff) > 0: 
                    self.warmup = True
                    self.decrease_rate = 0
                    self.increase_rate = 1
                    return True
        return False
        '''
        #print(pd.DataFrame(util.to_numpy_matrix(self.labels)))
            if len(self.prev) < 1:
                self.prev = np.sum(util.to_numpy_matrix(self.labels).tolist(), axis=0) / len(self.labels)
                self.labels = []
            else:
                r = np.sum(util.to_numpy_matrix(self.labels).tolist(), axis=0) / len(self.labels)
                print(r - self.prev)
                print(np.flip(np.argsort(r-self.prev))[:self.k])
                self.prev = r
                self.labels = []
        return False'''
        '''if len(self.history) < self.detection_window_size:
            r = self.recip_rank(util.to_numpy_matrix(self.labels)).tolist()[:self.k]
            prev = r
            self.add_to_history(r)
        else:
            r = self.recip_rank(util.to_numpy_matrix(self.labels)).tolist()[:self.k]
            rep = np.unique(self.history).tolist()
            self.add_to_history(r)
            if len(rep) > self.k and self.warmup == True:
                None
                print('Warmup noise')
            elif len(rep) > self.k and self.warmup == False:
                for l in r:
                    if l not in rep:
                        self.warmup = True
                        self.labels = []
                        return True
            else:
                if self.warmup:
                    print('Warmup has ended')
                    self.warmup = False
        return False'''
