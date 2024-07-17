import numpy as np
import pandas as pd
import math

from utils import shuffle_data
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.utils import TimeSeries


class BatchDataGenerator:
    def __init__(self, flags, stage, data, labels=[], n_past=1):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')
        self.data = data
        self.stage = stage
        self.n_past = n_past

        self.iter = flags.inner_loops
        self.feat_num = flags.feat_num
        self.batch_size = flags.batch_size
        self.batch_num = math.ceil(len(self.data)/flags.batch_size)
        self.labels = np.asfarray(np.eye(2)[labels])
    

    def get_batches(self, shuffle=False, seed=0): 
        # 
        if shuffle: 
            data, labels = shuffle_data(self.data, self.labels, seed=seed)
        else: 
            data, labels = self.data, self.labels
        
        # 
        if self.stage in ['train', 'val']: 
            data_lab = data 
            cols = ['value-' + str(i) for i in range(self.feat_num)]
        else: 
            data_lab = np.concatenate([data, labels], axis=1)
            cols = ['value-' + str(i) for i in range(self.feat_num)] + ['label-' + str(i) for i in range(2)]
        
        data_ts = TimeSeries.from_pd(pd.DataFrame(data_lab, columns=cols))  
        loader = RollingWindowDataset(
                    data_ts,
                    target_seq_index=None,
                    shuffle=shuffle,
                    flatten=True,
                    n_past=self.n_past,
                    n_future=0,
                    batch_size=self.batch_size,
                )
        
        if self.stage in ['train', 'val']:
            self.batch_data = [b[0] for b in loader]
        else: 
            col_num = self.feat_num + 2
            self.batch_data = [np.concatenate([b[0][:, col_num*i: col_num*(i+1)][:, :-2]
                                                for i in range(self.n_past)], axis=1) for b in loader]
            self.batch_labels = [b[0][:, -2:] for b in loader]
