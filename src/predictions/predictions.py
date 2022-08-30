import os
import numerapi
import numpy as np
import pandas as pd
from pathlib import Path

class Predicitons:
    def __init__(self, publ_id, priv_key, model_id):
        os.chdir('/Users/niklaskampe/Desktop/Coding/Numerai_Models/')
        self.pred_dir = ""
        self.cwd = os.getcwd()
        self.model_id = model_id
        self.api = numerapi.NumerAPI(publ_id, priv_key)
    def create_prediction_directory(self, model_name):
        try:
            current_round = self.api.get_current_round()
        except:
            print("Error in API Call.")
        self.pred_dir = f"./predictions/Round_{current_round}/{model_name}"
        if os.path.exists(self.pred_dir) == False:
            Path(self.pred_dir).mkdir(parents = True)
    def create_submission_file(self, df_pred, file_name, model_name):
        self.create_prediction_directory(model_name)
        os.chdir(self.pred_dir)
        df_pred.to_csv(file_name, index = False)  
        os.chdir(self.cwd)
        print(f"Submission File '{file_name}' Successfully Created.")
    def submit_predictions(self):
        os.chdir(self.pred_dir)
        self.api.upload_predictions("predictions.csv", model_id = self.model_id)
        os.chdir(self.cwd)
        print("Predictions Successfully Submitted.")     