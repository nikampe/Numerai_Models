import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import Objects
from src.data.data_import import Data
from data.feature_engineering import DataProcessed
from src.models.linear_models import LinearModels
from src.models.nonlinear_models import NonlinearModels
from src.predictions.predictions import Predicitons

# Static Variables
PUBL_ID = "XXXXXXXXXXXXXX"
PRIV_KEY = "XXXXXXXXXXXXXX"
MODEL_ID = "XXXXXXXXXXXXXX" # MAIN_MODEL_NLK 

def correlation(pred, target):
    ranked_pred = pred.rank(pct = True, method = "first")
    corr = np.corrcoef(ranked_pred, target)[0, 1]
    return corr

def score(pred, target):
    score = correlation(pred, target)
    return score

def model_evaluation(data):
    pred = data["prediction"]
    target = data["target"]
    eval_score = score(pred, target)
    return eval_score

def main():
    # Data Import & Preprocessing
    data = Data()
    data_train, data_test, data_live = data.import_data()
    data_processed = DataProcessed(data_train, data_test, data_live)
    data_train_processed, data_test_processed, data_live_processed = data_processed.preprocess()
    # Model/Class Initiation
    linear_models = LinearModels(data_train_processed, data_test_processed, data_live_processed)
    nonlinear_models = NonlinearModels(data_train_processed, data_test_processed, data_live_processed)
    # Prediction Generation
    model_name, df_pred_train, df_pred_test, df_pred_live = nonlinear_models.decision_tree(neutralizing = True)
    # In-Sample Model Evaluation
    eval_score_is = model_evaluation(df_pred_train)
    print(f"In-Sample (IS) Correlation Score: {eval_score_is}")
    df_pred_train.drop(["target"], axis = 1, inplace = True)
    # Out-of-Sample Model Evaluation
    eval_score_oos = model_evaluation(df_pred_test)
    print(f"Out-of-Sample (OOS) Correlation Score: {eval_score_oos}")
    df_pred_test.drop(["target"], axis = 1, inplace = True)
    # Final Prediction File
    predictions = Predicitons(PUBL_ID, PRIV_KEY, MODEL_ID)
    predictions.create_submission_file(df_pred_test, f"predictions_test.csv", model_name)
    predictions.create_submission_file(df_pred_live, f"predictions.csv", model_name)
    # Final Prediction Submission
    predictions.submit_predictions()

if __name__ == "__main__":
    main()

