import numpy as np
import pandas as pd
from datetime import datetime

from merlion.utils import TimeSeries
from merlion.models.factory import ModelFactory #  initialize models using the model factory
from merlion.post_process.threshold import AggregateAlarms
from metrics.eval_metrics import *


class MerlionBaseline:
    def __init__(self, flags):
        self.data_path = flags.data_path #flags.data_root + '_'.join(['tr_' + '_'.join([i[3:] for i in flags.train_data]), flags.test_data])
        self.dataset = flags.dataset
        self.sel_feats = ['feat_'+str(i) for i in range(flags.feat_num)]
        self.model_name = flags.model_name
        self.load_data() # Load the training and test data
        

    def load_data(self): 
        # Load the training data and convert to time-series
        tr_data = np.load(self.data_path + '/' + self.dataset + '_train.npy') 
        self.train_ts = TimeSeries.from_pd(pd.DataFrame(tr_data, columns=self.sel_feats))

        # Load the test data and convert to time-series
        te_data = np.load(self.data_path + '/' + self.dataset + '_test.npy')
        self.test_ts = TimeSeries.from_pd(pd.DataFrame(te_data, columns=self.sel_feats))

        # Load the test labels
        self.te_label = np.load(self.data_path + '/' + self.dataset + '_test_label.npy')


    def train(self): 
        # Train the AD model
        tr_start = datetime.now() 
        self.model = ModelFactory.create(self.model_name, threshold=AggregateAlarms(alm_threshold=2))

        # Get the anomaly score
        self.model.train(train_data=self.train_ts)

        tr_end = datetime.now()
        tr_time = round((tr_end - tr_start).total_seconds()/60, 4)
        print('Training time (min): ', tr_time)
        return tr_time


    def test(self): 
        # Get the training anomaly score
        self.tr_scores = np.array(self.model.get_anomaly_label(self.train_ts).to_pd()['anom_score'])

        # Get the test anomaly score
        te_start = datetime.now() 
        self.te_scores = np.array(self.model.get_anomaly_score(self.test_ts).to_pd()['anom_score'])

        te_end = datetime.now()
        te_time = round((te_end - te_start).total_seconds()/60, 4)
        print('Test time (min): ', te_time)

        # Get the overall evaluation results
        results = overall_evaluation(self.tr_scores, self.te_scores, self.te_label)
        return results, te_time