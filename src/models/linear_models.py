from email.mime import base
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
import warnings
warnings.filterwarnings('ignore')

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from data.data_import import Data
from models.utils import grid_search_cross_validation
from utils.helpers import feature_target_extraction, feature_extraction, create_pred_df

class LinearModels:
    def __init__(self, data_train, data_validation, data_live):
        self.data_train, self.data_validation, self.data_live = data_train, data_validation, data_live
        self.X_train, self.y_train = feature_target_extraction(self.data_train)
        self.X_validation, self.y_validation = feature_target_extraction(self.data_validation)
        self.X_live = feature_extraction(self.data_live)
        self.model_name = ""

    def ols_regression(self, intercept = True):
        self.model_name = "OLS"
        # In-/Exclude Intercept
        if intercept == True:
            self.X_train = np.hstack([np.ones(len(self.X_train))[:, np.newaxis], self.X_train])
            self.X_validation = np.hstack([np.ones(len(self.X_validation))[:, np.newaxis], self.X_validation])
            self.X_live = np.hstack([np.ones(len(self.X_live))[:, np.newaxis], self.X_live])
        # Model Fit
        coef = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        # Model Prediction - Train Data
        pred_train = self.X_train @ coef
        df_pred_train = create_pred_df(self.data_train, pred_train)
        df_pred_train["target"] = self.y_train
        # Model Prediction - Test Data
        pred_validation = self.X_validation @ coef
        df_pred_validation = create_pred_df(self.data_validation, pred_validation)
        df_pred_validation["target"] = self.y_validation
        # Model Prediction - Live Data
        pred_live = self.X_live @ coef
        df_pred_live = create_pred_df(self.data_live, pred_live)
        return self.model_name, df_pred_train, df_pred_validation, df_pred_live

    def ridge_regression(self):
        self.model_name = "Ridge"
        # Model Specification
        params = {"alpha": np.arange(0, 1, 0.05)}
        model = Ridge()
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {}
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(data_train, model, params)
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

    def lasso_regression(self):
        self.model_name = "Lasso"
         # Model Specification
        params = {"alpha": np.arange(0, 1, 0.05)}
        model = Lasso()
        # Hyperparameter Tuning (Penalty Parameter Alpha)
        best_params = {}
        if bool(best_params) == False:
            best_params = grid_search_cross_validation(data_train, model, params)
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

if __name__ == "__main__":
    data = Data()
    data_train, data_test, data_live = data.import_data()
    linear_models = LinearModels(data_train, data_test, data_live)
    linear_models.ridge_regression()

