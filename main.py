import sys
import numpy as np
import pandas as pd 
from skmultiflow.data import DataStream, MultilabelGenerator, ConceptDriftStream
from skmultiflow.meta import ClassifierChain
from sklearn.linear_model import SGDClassifier
from skmultiflow.metrics import hamming_score
from sklearn.metrics import accuracy_score, f1_score
from ld3 import LD3
from skmultilearn.dataset import load_from_arff
from sklearn.preprocessing import MultiLabelBinarizer
from skmultiflow.drift_detection import ADWIN, EDDM, KSWIN, HDDM_W, HDDM_A, DDM, PageHinkley
import util
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

'''datasets = [('Imdb.arff', 28, 3, 3000, 120, True, 20000, False),('Synthetic', 40, 5, 1500, 100, True, 4300, True), ('Synthetic', 40, 5, 1500, 100, True, 4300, False),
           ('Reuters-K500.arff', 103, 7, 200, 100, True, 100, False), ('OHSUMED-F.arff', 23, 6, 1100, 100, False, 3000, False), ('Stackex_coffee.arff', 123, 10, 60, 5, False, 500, False),
           ('Enron.arff', 53, 7, 200, 50, False, 2000, False), ('Yahoo_Health.arff', 32, 4, 1000, 100, False, 20000, False), ('Bibtex.arff', 159, 3, 2500, 75, False, 20000, False),
           ('tmc2007-500.arff', 22, 6, 2500, 100, True, 10000, False), ('Mediamill.arff', 101, 20, 3000, 100, True, 10000, False), ('20NG.arff', 20, 2, 2000, 100, True, 10000, False),
           ('Slashdot.arff', 22, 3, 750, 40, False, 20000, False)]'''
datasets = [('Synthetic', 20, 1, 500, 500, True)]#('Scene.arff', 6, 2, 100, 100, False)]
ks = []
for dataset, label_count, k, pretrain_size, window_size, sudden in datasets:
    print('Testing: {} with k={}'.format(dataset, k))
    if dataset == 'Synthetic':
        sample_size = 9000
        n_features = 100
        if sudden:
            print('Sudden')
            width = 1
        else:
            print('Gradual')
            width = 4000
        s1 = MultilabelGenerator(n_samples=sample_size, n_features=n_features, n_targets=label_count, n_labels=1, random_state=0)
        s2 = MultilabelGenerator(n_samples=sample_size, n_features=n_features, n_targets=label_count, n_labels=2, random_state=250)
        stream = ConceptDriftStream(stream=s1, drift_stream=s2, position=6000, width=width, random_state=0)
        #stream = s1
        X = np.zeros((sample_size, n_features))
        y = np.zeros((sample_size, label_count))
    else:  
        X, y = load_from_arff('./datasets/{}'.format(dataset), label_count=label_count)
        X = X.toarray()
        y = y.toarray().astype(np.int8)
        sample_size = len(X)
        num_features = X.shape[1]

        if len(np.unique(y)) > 2:
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)

        label_count = y.shape[1]
        stream = DataStream(data = X, y=y, n_targets=label_count)

    #detectors = [LD3(k=k, window_size=window_size), ADWIN(), DDM(), EDDM(), PageHinkley(), HDDM_A(), HDDM_W(), KSWIN(), None]
    #detector_names = ['LD3', 'ADWIN', 'DDM', 'EDDM', 'PageHinkley', 'HDDM_A', 'HDDM_W', 'KSWIN', 'No Detector']
    detectors = [LD3(k=k, window_size=window_size)]
    detector_names = ['LD3']
    clfs = [ClassifierChain(SGDClassifier(n_jobs=-1, loss='hinge',  random_state=1, warm_start=True), random_state=1) for det in detectors]
    pre_sample = [np.zeros(X.shape[1]), np.zeros(X.shape[1])]
    pre_label = [np.ones(y.shape[1]), np.zeros(y.shape[1])]

    for clf in clfs:
        clf.fit(np.array(pre_sample), np.array(pre_label))



    count = 0
    predicts_list = [[] for det in detectors]
    true_labels = []
    acc = [[] for det in detectors]
    hamms = [[] for det in detectors]
    drifts = [False for det in detectors]
    pretrain_X = []
    pretrain_y = []

    while stream.has_more_samples():
        if count == 6000:
            print('6000 Here')
        for i, drift in enumerate(drifts):
            if drift:
                ks.append(k)
                print('Drift detected with {} at: {}'.format(detector_names[i], count))
                clfs[i].reset()
                pretrain_X.insert(0, pre_sample[1])
                pretrain_X.insert(0, pre_sample[0])
                pretrain_y.insert(0, pre_label[1])
                pretrain_y.insert(0, pre_label[0])
                clfs[i].fit(np.array(pretrain_X), np.array(pretrain_y))
                drifts[i] = False

        try:
            sample, label = stream.next_sample()
            if dataset == 'Synthetic':
                util.add_to_pretrain(pretrain_X, pretrain_y, sample[0], label, pretrain_size)
                label = np.array([label]).astype(np.int32)
            else:
                util.add_to_pretrain(pretrain_X, pretrain_y, sample[0], label[0], pretrain_size)
            count += 1
        except:
            print(count)
            break

        for i, clf in enumerate(clfs):
            pred = clf.predict(sample)
            clf.partial_fit(sample, label)
            if detector_names[i] == 'LD3':
                detectors[i].add_element(pred.astype(np.int32))
            elif detector_names[i] == 'KSWIN':
                detectors[i].add_element(sample[0][0])
            elif detector_names[i] == 'No Detector':
                drift = False
            else:
                detectors[i].add_element((pred.astype(np.int32).flatten().tolist())==(label.astype(np.int32).flatten().tolist()))
            predicts_list[i].extend(pred)
 
        true_labels.extend(label)


        for i, detector in enumerate(detectors):
            if detector != None:
                drifts[i] = detector.detected_change()

        print('Stream progress: {}/{}'.format(count, sample_size), end='\r', flush=True)
    print('Stream progress: {}/{}'.format(count, sample_size))
    '''graph = detectors[0].debug
    cnt = detectors[0].count
    labels = [39, 8, 7, 17, 21, 15, 36, 19, 30]

    for label in labels:
        plt.plot(np.arange(cnt), graph[label])
    plt.legend(loc='upper left', labels=labels)
    plt.savefig('./label_prog.png')'''

    predicts_list = np.array(predicts_list)
    true_labels = np.array(true_labels)

    '''for i, detector_name in enumerate(detector_names):
        if dataset == 'Synthetic':
            if sudden:
                filename = './outputs/predicts/{}_{}_sudden.csv'.format(dataset, detector_name)
            else:
                filename = './outputs/predicts/{}_{}_incremental.csv'.format(dataset, detector_name)
        else:
            filename = './outputs/predicts/{}_{}.csv'.format(dataset, detector_name)
        df = pd.DataFrame(predicts_list[i], columns=['l{}'.format(i) for i in range(label_count)])
        df.to_csv(filename, index=False)
    
    df = pd.DataFrame(true_labels, columns=['l{}'.format(i) for i in range(label_count)])
    df.to_csv('./outputs/true_labels/{}.csv'.format(dataset), index=False)'''
    

    for i, predicts in enumerate(predicts_list):
        print('Accuracy of {}: {}'.format(detector_names[i], accuracy_score(true_labels, predicts)))
        print('Hamming score of {}: {}'.format(detector_names[i], hamming_score(true_labels, predicts)))
        print('F1-ex of {}: {}'.format(detector_names[i], f1_score(true_labels, predicts, average='samples')))
        print('F1-micro of {}: {}'.format(detector_names[i], f1_score(true_labels, predicts, average='micro')))
        print()