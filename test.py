import numpy as np
import util
from scipy import stats
from river.base import DriftDetector
from scipy.spatial import distance
import time


class LD3(DriftDetector):
    def __init__(self, k=2, window_size = 250, correlation_delta=0.05, relax_threshold=0.2, max_window_size=1000):
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.max_window_size = max_window_size
        self.initial_window_size = window_size
        self.w1 = Window(max_size=window_size)
        self.w2 = Window(max_size=window_size)
        self.w3 = Window(max_size=window_size)
        self.warmup = True
        self.warmup_count = window_size * 2
        self.r1 = None
        self.r2 = None
        self.correlation_delta = correlation_delta
        self.past_correlation = None
        self.relax_threshold = relax_threshold
        self.cum_sum = 0
        self.count = 0
        self.average_slope = True
        self.max_dist = 0

    def update(self, value):
        self.insert_element(value.flatten())
        if not self.warmup:
            self.r1 = util.recip_rank(util.to_numpy_matrix(self.w1.get_window))
            self.r2 = util.recip_rank(util.to_numpy_matrix(self.w2.get_window))
            self.count += 1

            #drift, warning = self.detect_change()
            drift, warning = False, False
            correlation = self.rank_correlation()
            self.w3.queue(self._average_correlation - correlation)
            self.cum_sum += correlation
    
            if stats.wilcoxon(self.w3.get_window)[1] < 0.05: #correlation > self.correlation_delta:
                drift = True
                warning = True
                
            if drift:
                self.clear_windows()
                #self.increase_windows(self.window_size*2)
                self.warmup_counter(warmup_count=self.window_size*2)
                print('Correlation: ', correlation)
                print('Average correlation: ', self._average_correlation)
                #self.cum_sum = 0
                #self.count = 0
            
            return drift, warning

        return False, False

    def rank_correlation(self, dist=distance.cityblock):
        self.count += 1
        x = (np.argsort(self.r1)).astype(np.float32)
        y = (np.argsort(self.r2)).astype(np.float32)
        if self.max_dist == 0:
            maxx = np.arange(1,len(x)+1,1).astype(np.float)
            maxy = np.flip(np.arange(1,len(x)+1,1)).astype(np.float) 
            self.max_dist = dist(maxx, maxy, w=1/((maxy)**maxy))
        #c = np.array([(dist(x[i], y[i])-1) * (1/((2)**(y[i]*y[i]))) for i in range(len(x))]).sum()
        #c = dist(x, y, w=1/((2)**(y*y))) / np.sqrt(len(x))
        c = dist(x, y, w=1/((y)**(y))) / self.max_dist
        self.cum_sum += c
        return c 

    def dist(self, x, y, w):

        sum_ = 0
        for i in range(len(x)):
            abs_ = np.abs(x[i] - y[i])
            sum_ +=  abs_*w[i] if abs_ > 1 else 0
        return sum_
        

    def warmup_counter(self, warmup_count=None):
        if self.warmup_count > 0:
            self.warmup_count -= 1
        else:
            self.warmup = False

        if warmup_count is not None:
            self.warmup_count = warmup_count
            self.warmup = True

    def insert_element(self, value):
        self.warmup_counter()
        v = self.w1.queue(value)
        if v is not None:
            u = self.w2.queue(v)
    
    def clear_windows(self):
        self.w1.clear()
        self.w2.clear()
        self.w3.clear()

    def increase_windows(self, value):
        if value < self.max_window_size:
            self.window_size = value
            self.w1.increase_size(value)
            self.w2.increase_size(value) 

    def decrease_windows(self, value):
        if value >= self.initial_window_size:
            self.window_size = value
            self.w1.decrease_size(value)
            self.w2.decrease_size(value) 

    def detect_change(self):
        # EXPERIMENT WITH:
        #
        # KENDALL TAU
        # WEIGHTED TAU
        # SPEARMANR
        # PEARSONR
        # SOMERS'D
        # GOODMAN AND KRUSKAL'S GAMMA

        warning_margin = self.k // 2
        r1 = self.r1[:self.k]
        r2 = self.r2[:self.k]
        for rank in r2:
            if rank not in r1:
                warning_margin -= 1
        
        return warning_margin < 0, False
    
    @property
    def _ranks(self):
        return self.r1, self.r2
    
    @property
    def _average_correlation(self):
        return self.cum_sum / self.count if self.count > 0 else 0


    def aptau(self, l1, l2):
        ci_sum = 0
        for i in range(1,len(l1)):
            item_x = l1[i]
            l_x = l1[:i]
            l_y = l2[:np.where(l2==item_x)[0][0]]
            ci = 0
            for j in range(len(l_x)):
                if l_x[j] in l_y:
                    ci += 1
            ci_sum += (ci/i)
        return ((2 * ci_sum) / (len(l2) - 1)) - 1
    
    def symmaptau(self, l1, l2):
        return (self.aptau(l1, l2) + self.aptau(l2, l1)) / 2

class Window():
    def __init__(self, max_size=250):
        self.max_size = max_size
        self.size = 0
        self.window = []
    
    def queue(self, value):
        if self.size <= self.max_size:
            self.window.append(value)
            self.size += 1
            popped = None
        else:
            popped = self.dequeue()
            self.window.append(value)
        return popped
    
    def dequeue(self):
        return self.window.pop(0)

    def clear(self):
        self.size = 0
        self.window.clear()

    def increase_size(self, value):
        self.max_size = value
    
    def decrease_size(self, value):
        self.max_size = value
    
    @property
    def get_window(self):
        return np.array(self.window)