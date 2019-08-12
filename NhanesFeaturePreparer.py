"""
This script facilitates the loading, feature selection, and
feature engineering of the analysis of the NHANES dataset.
"""
import sys
import numpy as np
import pandas as pd
import json
import nhanes as nhanes
from sklearn.feature_selection import mutual_info_classif

class NhanesFeaturePreparer:
     def __init__(self,raw_data: pd.DataFrame=None, training: bool=False, config_outpath, *, config_inpath):
         with open(config_inpath) as f:
             config = json.load(f)
         if training:
             self.DATA_PATH = config['DATA_PATH']
             self.DATASET = config['DATASET']
             self.SEED = config['SEED']
             self.TEST_SIZE = config['TEST_SIZE']
             self.TRAIN_SIZE = config['TRAIN_SIZE']
             self.FILTER_THRES = config['FILTER_THRES']
         else:
             self.VARS_TO_KEEP = config['VARS_TO_KEEP']

         self.raw_data = raw_data
         self.training = training
         self.config_outpath = config_outpath
         self.features = None
         self.target = None
         self.prepared_data = None
         self.X_train = None
         self.X_test = None
         self.y_train = None
         self.y_test = None

     def load_data(self):
         ds = nhanes.Dataset(self.DATA_PATH)
         ds.load_arthritis()
         self.features = ds.features
         self.target = ds.targets

     def split_data(self):
         np.random.seed(42)
         perm = np.random.permutation(self.target.shape[0])
         self.features = self.features[perm]
         self.target = self.target[perm]
         n_samples = self.features.shape[0]
         n_classes = int(self.target.max() + 1)

         test_idxs = np.random.permutation(np.arange(0,int(n_samples*0.15),1))
         train_idxs = np.random.permutation(np.arange(0,int(n_samples*0.30),1))

         test_batch_idxs = []
         train_batch_idxs = []
         for cl in range(n_classes):
             test_cl_idxs = test_idxs[self.target[test_idxs] == cl]
             train_cl_idxs = train_idxs[self.target[train_idxs] == cl]
             test_batch_idxs.extend(test_cl_idxs[:self.TEST_SIZE//n_classes])
             train_batch_idxs.extend(train_cl_idxs[:self.TRAIN_SIZE//n_classes])
         test_batch_idxs = np.random.permutation(test_batch_idxs)
         train_batch_idxs = np.random.permutation(train_batch_idxs)

         self.X_train = pd.DataFrame(self.features[train_batch_idxs])
         self.y_train = self.target[train_batch_idxs]
         self.X_test = pd.DataFrame(self.features[test_batch_idxs])
         self.y_test = self.target[test_batch_idxs]

     def correlation_filter(self):
         X_corr = self.X_train.corr()
         to_drop = []

         for i in range(len(X_corr)):
            for j in range(len(X_corr)):
                if (i != j) and (X_corr.iloc[i,j]>np.percentile(X_corr,self.FILTER_THRES)):
                    if ([i,j] not in to_drop) and ([j,i] not in to_drop):
                        to_drop.append([i,j])

        idxs = []
        for i in to_drop:
            if (i[0] not in idxs) and (i[1] not in idxs):
                idxs.append(i[1])

        self.X_train.drop(idxs,axis=1,inplace=True)
        self.X_test.drop(idxs,axis=1,inplace=True)

    def mutual_info_filter(self):
        mi = mutual_info_classif(self.X_train,self.y_train)
        mi_thres = np.percentile(mi,self.FILTER_THRES)
        to_keep = (mi > mi_thres)

        self.X_train = self.X_train.loc[:,to_keep]
        self.X_test = self.X_test.loc[:,to_keep]

    def var_filter(self):
        bin_idxs = []
        cont_idxs = []
        for i in self.X_train.columns:
            if self.X_train.loc[:,i].nunique() <= 2:
                bin_idxs.append(i)
            else:
                cont_idxs.append(i)

        X_train_bin = self.X_train.loc[:,bin_idxs]
        X_train_cont = self.X_train.loc[:,cont_idxs]

        bin_var = X_train_bin.var()
        bin_cut = np.percentile(bin_var,self.FILTER_THRES)
        X_train_bin = X_train_bin.loc[:,(bin_var>bin_cut)]

        cont_var = X_train_cont.var()
        cont_cut = np.percentile(cont_var,self.FILTER_THRES)
        X_train_cont = X_train_cont.loc[:,(cont_var>cont_cut)]

        self.X_train = pd.concat([X_train_bin,X_train_cont],axis=1)
        self.X_test = self.X_test[:,self.X_train.columns]

    def write_vars_to_keep(self):
        config = {}
        config['VARS_TO_KEEP'] = self.X_train.columns

        with open(self.config_outpath, 'w') as f:
            json.dump(config,f)

    def prepare_data(self):

        self.prepared_data = self.raw_data.copy(deep=True)

        if self.training:
            self.load_data()
            self.split_data()
            self.correlation_filter()
            self.mutual_info_filter()
            self.var_filter()
            self.write_vars_to_keep()
        else:
            self.prepared_data = self.prepared_data.loc[:,self.VARS_TO_KEEP]
