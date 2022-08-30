from logging import NullHandler
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from models.utils import grid_search_cross_validation, neutralization, risk_features
from utils.helpers import feature_target_extraction, feature_extraction, create_pred_df

class NonlinearModels:
    def __init__(self, data_train, data_test, data_live):
        self.data_train, self.data_test, self.data_live = data_train, data_test, data_live
        self.X_train, self.y_train = feature_target_extraction(self.data_train)
        self.X_test, self.y_test = feature_target_extraction(self.data_test)
        self.X_live = feature_extraction(self.data_live)
        self.model_name = ""

    def knn(self):
        self.model_name = "KNN"
        data_train = self.data_train
        X_train, y_train = self.X_train, self.y_train 
        X_test, y_test = self.X_test, self.y_test 
        X_live = self.X_live
        # Model Specification
        params = {
            "n_neighbors": [2, 5, 10, 15, 20, 30], 
            "weights": ["uniform", "distance"]}  
        model = KNeighborsRegressor()
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {}
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(data_train, model, params)
        model.set_params(**best_params[0])       
        # Model Fit
        model.fit(X_train, y_train)
        # Model Prediction - Train Data
        pred_train = model.predict(X_train)
        df_pred_train = create_pred_df(self.data_train, pred_train)
        df_pred_train["target"] = y_train
        # Model Prediction - Test Data
        pred_test = model.predict(X_test)
        df_pred_test = create_pred_df(self.data_test, pred_test)
        df_pred_test["target"] = y_test
        # Model Prediction - Live Data
        pred_live = model.predict(X_live)
        df_pred_live = create_pred_df(self.data_live, pred_live)
        return self.model_name, df_pred_train, df_pred_test, df_pred_live
    
    def decision_tree(self, neutralizing = False):
        self.model_name = "Tree"
        features = [col for col in self.data_train if col.startswith("feature")]
        riskiest_features = risk_features(self.data_train, n = 50)
        X_train, y_train = self.X_train, self.y_train 
        X_test, y_test = self.X_test, self.y_test 
        X_live = self.X_live
        # Model Specification
        params = {
            "max_depth": [5, 10, 20], # Max depth of the tree -->
            "min_samples_split": [50, 100, 500], # Min number of samples required to split internal node -->
            "min_samples_leaf": [50, 100, 500], # Min number of samples required at a leaf node --> 
            "max_features": ["log2", "sqrt"]} # Min number of features considered for best split  --> "log2"
        model = DecisionTreeRegressor(criterion = "squared_error", splitter = "best")
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {} # Min number of features considered for best split  --> "log2"
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(self.data_train, model, params, neutralizing = neutralizing)
        model.set_params(**best_params[0])       
        # Model Fit
        model.fit(X_train, y_train)
        # Model Prediction - Train Data
        pred_train = model.predict(X_train)
        if neutralizing == True:
            pred_train = neutralization(data = self.data_train, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_train, normalize = True)
            self.data_train["prediction"] = pred_train
            self.data_train["prediction"] = self.data_train["prediction"].rank(pct = True)
        df_pred_train = self.data_train[["id", "prediction"]]
        df_pred_train["target"] = y_train
        # Model Prediction - Test Data
        pred_test = model.predict(X_test)
        if neutralizing == True:
            pred_test = neutralization(data = self.data_test, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_test, normalize = True)
            self.data_test["prediction"] = pred_test
            self.data_test["prediction"] = self.data_test["prediction"].rank(pct = True)
        df_pred_test = self.data_test[["id", "prediction"]]
        df_pred_test["target"] = y_test
        # Model Prediction - Live Data
        pred_live = model.predict(X_live)
        if neutralizing == True:
            pred_live = neutralization(data = self.data_live, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_live, normalize = True)
            self.data_live["prediction"] = pred_live
            self.data_live["prediction"] = self.data_live["prediction"].rank(pct = True)
        # df_pred_live = create_pred_df(self.data_live, pred_live)
        df_pred_live = self.data_live[["id", "prediction"]]
        return self.model_name, df_pred_train, df_pred_test, df_pred_live

if __name__ == "__main__":
    nonlinear_models = NonlinearModels()
    nonlinear_models.knn()
       
    