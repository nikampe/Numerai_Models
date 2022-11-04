import csv
import numpy as np
import pandas as pd

def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype = dtypes)
    return df

def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    features, target = feature_target_extraction(df)
    df[np.concatenate(features, target)]
    return df

def feature_extraction(data):
    features = [col for col in data if col.startswith("feature")]
    X = data[features].to_numpy()
    return X

def feature_target_extraction(data):
    features = [col for col in data if col.startswith("feature")]
    X = data[features].to_numpy()
    y = data["target"].to_numpy()
    return X, y

def create_pred_df(data, pred):
    df_pred = pd.DataFrame(columns = ["id", "prediction"])
    df_pred["id"] = data["id"]
    df_pred["prediction"] = pred
    return df_pred

def unif(data):
    x = (data.rank(method = "first") - 0.5) / len(data)
    return pd.Series(x, index = data.index)