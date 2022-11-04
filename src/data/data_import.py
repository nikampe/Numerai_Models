import os
import sys
import numerapi
import numpy as np
import pandas as pd
from pathlib import Path

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from utils.helpers import read_csv, read_parquet

class Data:

    def __init__(self):
        os.chdir('/Users/niklaskampe/Desktop/Coding/Numerai_Models/')
        self.data_dir = ""
        self.current_round = ""
        self.cwd = os.getcwd()
        self.api = numerapi.NumerAPI()

    def create_data_directory(self):
        try:
            self.current_round = self.api.get_current_round()
        except:
            print("Error in API Call.")
        self.data_dir = f"./data/Round_{self.current_round}"
        if os.path.exists(self.data_dir) == False:
            Path(self.data_dir).mkdir(parents = False, exist_ok = True)

    def download_data(self):
        print("---------- Data Download ----------")
        self.create_data_directory()
        dir_list = os.listdir(self.data_dir)
        if len(dir_list) == 3:
            try:
                os.chdir(self.data_dir)
                # self.api.download_dataset(filename = 'v4/train.parquet', dest_path = "data_train.parquet", round_num = self.current_round)
                # self.api.download_dataset(filename = 'v4/validation.parquet', dest_path = "data_validation.parquet", round_num = self.current_round)
                # self.api.download_dataset(filename = 'v4/live.parquet', dest_path = "data_live.parquet", round_num = self.current_round)
                os.chdir(self.cwd)
            except:
                print("Error in API Call.")
        else:
            print("Data Sets already Downloaded.\n")

    def import_data(self):
        self.download_data()
        print("---------- Data Import ----------")
        print("Import Training Data ...")
        data_train = read_parquet(self.data_dir + "/data_train.parquet")
        print("Import Validation Data ...")
        data_validation = read_parquet(self.data_dir + "/data_validation.parquet")
        print("Import Live Data ...")
        data_live = read_parquet(self.data_dir + "/data_live.parquet")
        print("Data successfully Imported.\n")
        return data_train, data_validation, data_live

if __name__ == "__main__":
    data = Data()
    data.download_data()