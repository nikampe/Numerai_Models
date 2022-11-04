import os
import sys
import joblib
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
    def __init__(self, data_train, data_validation, data_live):
        self.data_train, self.data_validation, self.data_live = data_train, data_validation, data_live
        self.X_train, self.y_train = feature_target_extraction(self.data_train)
        self.X_validation, self.y_validation = feature_target_extraction(self.data_validation)
        self.X_live = feature_extraction(self.data_live)
        self.model_name = ""

    def knn(self):
        self.model_name = "KNN"
        # Model Specification
        params = {
            "n_neighbors": [2, 5, 10, 15, 20, 30], 
            "weights": ["uniform", "distance"]}  
        model = KNeighborsRegressor()
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {}
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(self.data_train, model, params)
        model.set_params(**best_params[0])       
        # Model Fit
        model.fit(self.X_train, self.y_train)
        # Model Prediction - Train Data
        pred_train = model.predict(self.X_train)
        df_pred_train = create_pred_df(self.data_train, pred_train)
        df_pred_train["target"] = self.y_train
        # Model Prediction - Test Data
        pred_validation = model.predict(self.X_validation)
        df_pred_validation = create_pred_df(self.data_validation, pred_validation)
        df_pred_validation["target"] = self.y_validation
        # Model Prediction - Live Data
        pred_live = model.predict(self.X_live)
        df_pred_live = create_pred_df(self.data_live, pred_live)
        return self.model_name, df_pred_train, df_pred_validation, df_pred_live
    
    def decision_tree(self, neutralizing = False, retrain = False):
        self.model_name = "Tree"
        features = [col for col in self.data_train if col.startswith("feature")]
        riskiest_features = risk_features(self.data_train, n = 50)
        # Model Specification
        model = DecisionTreeRegressor(criterion = "squared_error", splitter = "best")
        params = {
            "max_depth": [5, 8, 10], # Max depth of the tree
            "min_samples_split": [50, 100, 200], # Min number of samples required to split internal node
            "min_samples_leaf": [50, 100, 200], # Min number of samples required at a leaf node
            "max_features": ["log2", "sqrt"]} # Min number of features considered for best split
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {
            "max_depth": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 200,
            "max_features": "sqrt"
        }
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(self.data_train, model, params, neutralizing = neutralizing)
        model.set_params(**best_params)       
        # Load, Fit & Save Model
        if retrain == True:
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, "model_tree.sav")
        else:
            model = joblib.load("model_tree.sav") 
        # Model Prediction - Train Data
        pred_train = model.predict(self.X_train)
        if neutralizing == True:
            pred_train = neutralization(data = self.data_train, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_train, normalize = True)
        self.data_train["prediction"] = pred_train
        self.data_train["prediction"] = self.data_train["prediction"].rank(pct = True)
        df_pred_train = self.data_train[["id", "prediction"]]
        df_pred_train["target"] = self.y_train
        # Model Prediction - Validation Data
        pred_validation = model.predict(self.X_validation)
        if neutralizing == True:
            pred_validation = neutralization(data = self.data_validation, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_validation, normalize = True)
        self.data_validation["prediction"] = pred_validation
        self.data_validation["prediction"] = self.data_validation["prediction"].rank(pct = True)
        df_pred_validation = self.data_validation[["id", "prediction"]]
        df_pred_validation["target"] = self.y_validation
        # Model Prediction - Live Data
        pred_live = model.predict(self.X_live)
        if neutralizing == True:
            pred_live = neutralization(data = self.data_live, columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred_live, normalize = True)
        self.data_live["prediction"] = pred_live
        self.data_live["prediction"] = self.data_live["prediction"].rank(pct = True)
        # df_pred_live = create_pred_df(self.data_live, pred_live)
        df_pred_live = self.data_live[["id", "prediction"]]
        return self.model_name, df_pred_train, df_pred_validation, df_pred_live

if __name__ == "__main__":
    nonlinear_models = NonlinearModels()
    nonlinear_models.knn()
       
    