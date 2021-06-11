import numpy as np
from numpy.core.numeric import correlate
import util
from scipy import stats
from river.base import DriftDetector
from scipy.spatial import distance
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import time


class LD3(DriftDetector):
    def __init__(self, k=2, window_size = 250, correlation_thresh=0.05, bandwith=0.1, max_window_size=1000, correlation_coeff=5):
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.max_window_size = max_window_size
        self.initial_window_size = window_size
        self.w1 = Window(max_size=window_size)
        self.w2 = Window(max_size=window_size)
        self.w3 = Window(max_size=window_size)
        self.model = KernelDensity(bandwidth=bandwith, kernel='gaussian')
        self.warmup = True
        self.warmup_count = window_size * 2
        self.r1 = None
        self.r2 = None
        self.correlation_thresh = correlation_thresh
        self.past_correlation = None
        self.cum_sum = 0
        self.count = 0
        self.max_dist = 0
        self.prev_correlation = None
        self.correlation_coeff = correlation_coeff
        self.average_correlation = None

    def update(self, value):
        self.insert_element(value.flatten())
        if not self.warmup:
            self.r1 = util.recip_rank(util.to_numpy_matrix(self.w1.get_window, self_loops=False))
            self.r2 = util.recip_rank(util.to_numpy_matrix(self.w2.get_window, self_loops=False))


            drift, warning = False, False
            correlation= self.WS()
            self.w3.queue(correlation)

            if self.w3.len < self.window_size:
                return False, False, 0, 0

            self.model.fit((self.w3.get_window).reshape((self.w3.len, 1)))
            '''if self.average_correlation == None:
                self.average_correlation = correlation
            else:
                self.average_correlation = 0.875 * self.average_correlation + 0.125 * correlation'''

            score = np.exp(self.model.score([[correlation]]))
            if  score < self.correlation_thresh: #correlation < self.correlation_thresh: 
                drift = True
                warning = True
                
            '''if self.count >= 194 and self.count <=205:
                print('Correlation: ', correlation)'''

            if drift:
                values = np.asarray([value for value in np.linspace(-1,1,100)])
                values = values.reshape((len(values), 1))
                probabilities = self.model.score_samples(values)
                probabilities = np.exp(probabilities)
                # plot the histogram and pdf
                plt.hist(self.w3.get_window, bins=36, density=True)
                plt.plot(values[:], probabilities)
                plt.show()
                self.clear_windows()
                #self.increase_windows(self.window_size*2)
                self.warmup_counter(warmup_count=self.window_size*2)
                print('Correlation: ', correlation)
                print('Score: ', np.exp(self.model.score([[correlation]])))
 
            
            self.prev_correlation = correlation

            return drift, warning, correlation, score

        return False, False, 0, 0


    '''def rank_correlation(self,k=0.1):
        lenght = len(self.r1)*k
        set1 = self.r1[:int(np.ceil(lenght))]
        set2 = self.r2[:int(np.ceil(lenght))]
        in_ = np.isin(set1, set2).sum()
        out_ = lenght - in_
        return (in_ - out_) / lenght'''

    def WS(self):
        sum_ = 0
        ranks_x = np.argsort(self.r1)
        ranks_y = np.argsort(self.r2)
        for i in range(len(self.r1)):
            sum_ += (1/(2**(ranks_x[i]))) * ((np.abs(ranks_x[i] - ranks_y[i]))/(np.max([np.abs(1-ranks_x[i]), np.abs(len(self.r1) - ranks_x[i])])))
        return 1 - sum_

    def rank_correlation(self, dist=distance.cityblock):
        self.count += 1
        x = (np.argsort(self.r1)).astype(np.float)
        y = (np.argsort(self.r2)).astype(np.float)
        if self.max_dist == 0:
            maxx = np.arange(1,len(x)+1,1).astype(np.float)
            maxy = np.flip(np.arange(1,len(x)+1,1)).astype(np.float) 
            self.max_dist = dist(maxx, maxy, w=1/((maxy)**maxy))
        #c = np.array([(dist(x[i], y[i])-1) * (1/((2)**(y[i]*y[i]))) for i in range(len(x))]).sum()
        c1 = self.dist(x, y, w=1/((3)**(x*x))) / np.sqrt(len(x))
        c2 = self.dist(x, y, w=1/((3)**(y*y))) / np.sqrt(len(x))
        c = (c1 + c2) / 2
        #c = dist(x, y, w=1/((y)**(y))) / self.max_dist
        self.cum_sum += c
        return c #/ self._average_correlation
 

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
        #self.w2.set_window(self.w1.get_window.tolist())
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
    
    @property
    def _ranks(self):
        return self.r1, self.r2
    
    @property
    def _average_correlation(self):
        return self.average_correlation #self.cum_sum / self.count if self.count > 0 else 0


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
    
    def set_window(self, window):
        self.window = window
    
    @property
    def get_window(self):
        return np.array(self.window)
    
    @property 
    def len(self):
        return len(self.window)