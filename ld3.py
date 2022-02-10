import numpy as np
from numpy.core.numeric import correlate
from numpy.random import sample
import util
from scipy import stats
from river.base import DriftDetector
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time
from skmultiflow.data import DataStream, MultilabelGenerator, ConceptDriftStream


class LD3(DriftDetector):
    def __init__(self, window_size = 500, correlation_thresh=3, len=2):
        super().__init__()
        self.window_size = window_size
        self.len = len
        self.w1 = Window(max_size=window_size)
        self.w2 = Window(max_size=window_size)
        self.w3 = Window(max_size=window_size)
        self.warmup = True
        self.warmup_count = window_size * 2
        self.r1 = None
        self.r2 = None
        self.correlation_thresh = correlation_thresh
        self.past_correlation = None
        self.prev_correlation = None

    def update(self, value):
        self.insert_element(value.flatten())
        if not self.warmup:
            self.r1 = util.recip_rank(util.to_numpy_matrix(self.w1.get_window, self_loops=False), self_loops=False)
            self.r2 = util.recip_rank(util.to_numpy_matrix(self.w2.get_window, self_loops=False), self_loops=False)


            drift, warning = False, False
            correlation= self.WS()
            

            if self.w3.len < self.window_size:
                self.w3.queue(correlation)
                return False, False, 0, 0

            score = util.find_anomalies(self.w3.get_window, coeff=self.correlation_thresh)
            self.w3.queue(correlation)
            if len(score) > self.len and correlation < 1: 
                drift = True
                warning = True
                

            if drift:
                values = np.asarray([value for value in np.linspace(-1,1,100)])
                values = values.reshape((len(values), 1))
                self.clear_windows()
                self.warmup_counter(warmup_count=self.window_size*2)

 
            
            self.prev_correlation = correlation

            return drift, warning, correlation, score

        return False, False, 0, 0


    def WS(self):
        sum_ = 0
        ranks_x = np.argsort(self.r1)
        ranks_y = np.argsort(self.r2)
        for i in range(len(self.r1)):
            sum_ += (1/(2**(ranks_x[i]))) * ((np.abs(ranks_x[i] - ranks_y[i]))/(np.max([np.abs(1-ranks_x[i]), np.abs(len(self.r1) - ranks_x[i])])))
        return 1 - sum_
        

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

    
    @property
    def _ranks(self):
        return self.r1, self.r2
    


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

class StreamGenerator():
    def __init__(self):
        pass

    def get_stream(self, type):
        sample_size = 0
        stream = None
        n_features = 0
        n_targets = 0
        if type == 'sudden1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=25, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=25, random_state=0)
            sample_size=20000
        elif type == 'gradual1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=250, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=250, random_state=0)
            sample_size=20000
        elif type == 'sudden2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'sudden3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=50, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=50, random_state=0)
            sample_size=20000
        elif type == 'mixed':
            n_features = 100
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=25000, n_features=n_features, n_targets=n_targets, n_labels=1, random_state=100) 
            s2 = MultilabelGenerator(n_samples=25009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=250)
            s3 = MultilabelGenerator(n_samples=25015, n_features=n_features, n_targets=n_targets, n_labels=3, random_state=0)
            s4 = MultilabelGenerator(n_samples=25015, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100)
            s5 = MultilabelGenerator(n_samples=25009, n_features=n_features, n_targets=n_targets, n_labels=1, random_state=250)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=20000, width=1, random_state=0)
            stream2 = ConceptDriftStream(stream=stream1, drift_stream=s3, position=40000, width=100, random_state=0)
            stream3 = ConceptDriftStream(stream=stream2, drift_stream=s4, position=60000, width=100, random_state=0)
            stream = ConceptDriftStream(stream=stream3, drift_stream=s5, position=80000, width=1, random_state=0)
            sample_size=100000
        elif type == 'gradual2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        elif type == 'gradual3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1000, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1000, random_state=0)
            sample_size=20000
        elif type == 'reoccurring':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        elif type == 'benchmark1':
            n_features = 100
            n_targets = 20
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'benchmark2':
            n_features = 100
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'benchmark3':
            n_features = 100
            n_targets = 100
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        return stream, sample_size, n_features, n_targets