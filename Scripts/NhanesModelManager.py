"""
This script facilitates the model fitting and prediction for
the analysis of NHANES dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from NhanesFeaturePreparer import NhanesFeaturePreparer

class NhanesModelManager:
    def __init__(self, joblib_inpath: str=None, joblib_outpath: str=None,*,nhanes_feature_preparer: NhanesFeaturePreparer):
        self.processor = nhanes_feature_preparer
        self.joblib_inpath = joblib_inpath
        self.joblib_outpath = joblib_outpath
        self.trained_model = None

    def fit_model(self):
        gbdt = GradientBoostingClassifier(n_estimators=30,
                                          min_samples_split=0.2,
                                          random_state=self.processor.SEED)

        gbdt.fit(self.processor.X_train,self.processor.y_train)

        self.trained_model = gbdt

    def write_model(self):
        joblib.dump(self.trained_model,self.joblib_outpath)

    def load_model(self):
        self.trained_model = joblib.load(self.joblib_inpath)

    def run_pipeline(self):
        self.processor.prepare_data()
        if self.processor.training:
            self.fit_model()
            self.write_model()
            preds = self.trained_model.predict(self.processor.X_test)
        else:
            self.load_model()
            preds = self.trained_model.predict(self.processor.prepared_data)

        return preds


if __name__ == "__main__":
    processor = NhanesFeaturePreparer(training=True,
                                      config_outpath='config/sample_outpath.json',
                                      config_inpath='config/sample_inpath.json')

    manager = NhanesModelManager(nhanes_feature_preparer=processor,
                                 joblib_outpath='model/sample_outpath.jl')

    manager.run_pipeline()
