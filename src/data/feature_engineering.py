import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/')
from data.data_import import Data

class DataProcessed:
    def __init__(self, data_train, data_test, data_live):
        self.data_train = data_train
        self.data_test = data_test
        self.data_live = data_live

    def imputation(self, data):
        features = [col for col in data if col.startswith("feature")]
        for feature in features:
            if data[feature].isnull().sum() > 0:
                data[feature] = data[feature].fillna(data[feature].mean())
        return data
    
    def neutralization(self, n = 5):
        data_train = self.data_train
        # Inconsistent Features (Changing Correlations with Target over Time)
        features = [col for col in data_train if col.startswith("feature")]
        feature_corr = data_train.groupby("era").apply(lambda era: era[features].corrwith(era["target"]))
        feature_corr_sorted = feature_corr.index.sort_values()
        feature_corr_h1 = feature_corr_sorted[0:len(feature_corr_sorted)//2]
        feature_corr_h2 = feature_corr_sorted[len(feature_corr_sorted)//2:]
        feature_corr_h1_mean = feature_corr.loc[feature_corr_h1,:].mean()
        feature_corr_h2_mean = feature_corr.loc[feature_corr_h2,:].mean()
        feature_corr_diff = feature_corr_h2_mean - feature_corr_h1_mean
        n_inconsistent_features = feature_corr_diff.abs().sort_values(ascending = False).head(n).index.tolist()
        return n_inconsistent_features
        # Plot of Inconsistent Features over Time

    def preprocess(self):
        data_train_processed = self.imputation(self.data_train)
        data_test_processed = self.imputation(self.data_test)
        data_live_processed = self.imputation(self.data_live)
        # # Preprocessing Steps
        # inconsistent_features = self.neutralization(n = 5)
        # # Adjust Training Data 
        # data_train_processed = data_train.drop(columns = inconsistent_features)
        # data_test_processed = data_test.drop(columns = inconsistent_features)
        # data_live_processed = data_live.drop(columns = inconsistent_features)
        return data_train_processed, data_test_processed, data_live_processed

if __name__ == "__main__":
    data = Data()
    data_train, data_test, data_live = data.import_data()
    data_train_processed = DataProcessed(data_train, data_test, data_live)
    data_train_processed.preprocess()