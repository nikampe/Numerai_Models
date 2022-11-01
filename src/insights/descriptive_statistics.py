import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from utils.helpers import feature_target_extraction, feature_extraction, create_pred_df
from data.data_import import Data

def corr(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    corr = np.corrcoef(y_true, rank_pred)[0,1]
    return corr

class Statistics:
    def __init__(self, data_train, data_test, data_live):
        self.data_train, self.data_test, self.data_live = data_train, data_test, data_live
        self.X_train, self.y_train = feature_target_extraction(self.data_train)
        self.X_test, self.y_test = feature_target_extraction(self.data_test)
        self.X_live = feature_extraction(self.data_live)
    
    def summary_statistics(self):
        print("-------- Summary Statistics --------")
        data_train = self.data_train
        num_obs = data_train.shape[0]
        print("Number of Observations: ", num_obs)
        features = [col for col in data_train if col.startswith("feature")]
        num_features = len(features)
        print("Number of Features: ", num_features)
        data_train["era_int"] = data_train.era.str.slice(3).astype(int)
        num_unique_eras = len(data_train["era_int"].unique())
        print("Number of Unique Eras: ", num_unique_eras)
        min_era = data_train["era_int"].min()
        print("Minimum Era: ", min_era)
        max_era = data_train["era_int"].max()
        print("Maxmimum Era: ", max_era)
        feature_groups = {group: [col for col in data_train if col.startswith(f"feature_{group}")] for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]}
        for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
            num_obs_feature_group = len(feature_groups[group])
            print(f"Number of Features in '{group}': ", num_obs_feature_group) 

    def correlations(self):
        print("-------- Correlations --------")
        data_train = self.data_train
        features = [col for col in data_train if col.startswith("feature")]
        # Feature Correlations
        feature_corr = data_train[features].corr()
        feature_corr_stacked = feature_corr.stack()
        feature_corr_df = feature_corr_stacked[feature_corr_stacked.index.get_level_values(0) < feature_corr_stacked.index.get_level_values(1)]
        feature_corr_df = feature_corr_df.sort_values()
        print("Feature Correlations:\n", feature_corr_df)
        # Feature-Target Correlations
        data_train["era_int"] = data_train.era.str.slice(3).astype(int)
        feature_target_corr = {feature: corr(data_train["target"], data_train[feature], data_train["era_int"]) for feature in features}
        feature_target_corr_ser = pd.Series(feature_target_corr).sort_values()
        print("Feature-Target Correlations:\n", feature_target_corr_ser)

class Plots:
    def __init__(self, data_train, data_test, data_live):
        self.data_train, self.data_test, self.data_live = data_train, data_test, data_live
        self.X_train, self.y_train = feature_target_extraction(self.data_train)
        self.X_test, self.y_test = feature_target_extraction(self.data_test)
        self.X_live = feature_extraction(self.data_live)

    def features_over_time(self):
        data_train = self.data_train
        data_train["era_int"] = data_train.era.str.slice(3).astype(int)
        # Distribution of Eras
        eras = data_train["era"].unique()
        features = [col for col in data_train if col.startswith("feature")]
        for era in eras:
            data_train_plot = data_train[data_train["era"] == era]
            plt.plot(data_train_plot[features])
        plt.title("Features over Time | Training Data", size = 14)
        plt.xlabel("Eras", size = 12)
        plt.ylabel("Value", size = 12)
        # plt.legend()
        plt.show()

if __name__ == "__main__":
    data = Data()
    data_train, data_test, data_live = data.import_data()
    statistics = Plots(data_train, data_test, data_live)
    statistics.features_over_time()