import os
import numerapi
import numpy as np
import pandas as pd
from pathlib import Path

class Predicitons:
    
    def __init__(self, publ_id, priv_key):
        os.chdir('/Users/niklaskampe/Desktop/Coding/Numerai_Models/')
        self.pred_dir = ""
        self.cwd = os.getcwd()
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

    def submit_predictions(self, model_id):
        os.chdir(self.pred_dir)
        print(os.chdir)
        try:
            self.api.upload_predictions("predictions.csv", model_id = model_id)
            print("Predictions Successfully Submitted.")     
        except:
            "Error in API submission."
        os.chdir(self.cwd)