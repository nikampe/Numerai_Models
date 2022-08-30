import os
import sys
import numerapi
import numpy as np
import pandas as pd
from pathlib import Path

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from utils.helpers import read_csv

class Data:
    def __init__(self):
        os.chdir('/Users/niklaskampe/Desktop/Coding/Numerai_Models/')
        self.data_dir = ""
        self.file_dir = ""
        self.cwd = os.getcwd()
        self.api = numerapi.NumerAPI()
    def create_data_directory(self):
        try:
            current_round = self.api.get_current_round()
        except:
            print("Error in API Call.")
        self.data_dir = f"./data/Round_{current_round}"
        self.file_dir = self.data_dir + f"/numerai_dataset_{current_round}"
        if os.path.exists(self.data_dir) == False:
            Path(self.data_dir).mkdir(parents = False, exist_ok = True)
    def download_data(self):
        print("---------- Data Download ----------")
        self.create_data_directory()
        dir_list = os.listdir(self.data_dir)
        if len(dir_list) == 0:
            try:
                os.chdir(self.data_dir)
                self.api.download_current_dataset(unzip = True)
                os.chdir(self.cwd)
            except:
                print("Error in API Call.")
        else:
            print("Data Sets already Downloaded.\n")
    def import_data(self):
        self.download_data()
        print("---------- Data Import ----------")
        training_path = self.file_dir + "/numerai_training_data.csv"
        tournament_path = self.file_dir + "/numerai_tournament_data.csv"
        data_train = read_csv(training_path)
        data_tournament = read_csv(tournament_path)
        print("Data successfully Imported.\n")
        data_test = data_tournament[data_tournament.data_type == "validation"]
        data_live = data_tournament[data_tournament.data_type == "live"]
        # print("Train, Test & Live Data Set succesfully Imported.")
        # print("Train Data Set:\n", data_train.head(n = 5))
        # print("Test Data Set:\n", data_test.head(n = 5))
        # print("Live Data Set:\n", data_live.head(n = 5))
        return data_train, data_test, data_live

if __name__ == "__main__":
    data = Data()
    data.import_data()