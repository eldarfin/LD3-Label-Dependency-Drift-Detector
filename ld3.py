import numpy as np
import util

class DriftDetector():
    def __init__(self, k, label_window_size=100, detection_window_size=2, max_window_size=2, label_count=10000, sudden_drift=False, big_dataset=False):
        self.history = []
        self.detection_window_size = detection_window_size
        self.max_window_size = max_window_size
        self.k = k
        self.label_window_size = label_window_size
        self.labels = []
        self.warmup = True
        self.sudden_drift = sudden_drift
        self.label_count = label_count
        self.big_dataset = big_dataset

    def add_element(self, y):
        if self.big_dataset:
            if len(self.labels) < self.label_count:
                self.labels.extend(y)
            else:
                self.labels.pop(0)
                self.labels.extend(y)
        else:
            self.labels.extend(y)

    def recip_rank(self, mat):
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

    def add_to_history(self, item):
        if len(self.history) < self.detection_window_size:
            self.history.append(item)
        else:
            self.history.pop(0)
            self.history.append(item)

    def detected_change(self):
        if len(self.labels) % self.label_window_size == 0 and len(self.labels) != 0:
            if len(self.history) < self.detection_window_size:
                r = self.recip_rank(util.to_numpy_matrix(self.labels)).tolist()[:self.k]
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
            if self.sudden_drift:
                self.labels = []
            return False
