import sys
import numpy as np
import scipy as sp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import Objects
sys.path.insert(1, '/Users/niklaskampe/Desktop/Coding/Numerai_Models/src/') 
from utils.helpers import unif

def train_test_folds(data, k = 3, embargo = 12):
    eras = data["era"].unique()
    # Length of Each Fold
    fold_len = int(np.floor(len(eras) / k))
    # Test Folds
    folds_test = []
    for i in range(0, k):
        fold_test = eras[(i * fold_len):((i + 1) * fold_len)]
        folds_test.append(fold_test)
    if len(eras) % k != 0:
        folds_test[-1] = np.append(folds_test[-1], eras[-len(eras):])
    # Train Folds
    folds_train = []
    for fold_test in folds_test:
        fold_test_int = [int(i[3:]) for i in fold_test]
        fold_test_max = int(np.max(fold_test_int))
        fold_test_min = int(np.min(fold_test_int))
        eras_train = [] 
        for era in eras:
            if not (fold_test_min <= int(era[3:]) <= fold_test_max):
                eras_train.append(era)
        fold_train = []
        # Embargoing Train and Test Folds to Account for Overlapping Autoregressive Structures
        for era in eras_train:
            if (abs(int(era[3:]) - fold_test_max) > embargo) and (abs(int(era[3:]) - fold_test_min) > embargo):
                fold_train.append(era)
        folds_train.append(fold_train)
    return folds_train, folds_test

def risk_features(data, n = 50):
    features = [col for col in data if col.startswith("feature")]
    # Feature Correlations with Target
    corr = data.groupby("era").apply(lambda d: d[features].corrwith(d["target"]))
    corr_eras = corr.index.sort_values()
    # Correlations in First and Second Half of Eras
    corr_eras_h1 = corr_eras[:len(corr_eras) // 2]
    corr_eras_h2 = corr_eras[len(corr_eras) // 2:]
    # Mean Correlations in First and Second Half of Eras
    corr_h1_mean = corr.loc[corr_eras_h1, :].mean()
    corr_h2_mean = corr.loc[corr_eras_h2, :].mean()
    corr_diff = corr_h2_mean - corr_h1_mean
    # n Riskiest Features in terms of Changing Correlation over Eras
    riskiest_features = corr_diff.abs().sort_values(ascending = False).head(n).index.tolist()
    return riskiest_features

def neutralization(data, columns_to_neutralize, features_to_neutralize, pred, normalize = True):
    data["pred"] = pred
    eras = data["era"].unique()
    pred_neutralized = []
    for era in eras:
        data_era = data[data["era"] == era]
        # 3xn Matrix: Selected Feature Values for each Era
        per_era_values = data_era[columns_to_neutralize].values
        # nx3 Matrix: Selected Feature Values for all Eras
        per_feature_values = per_era_values.T
        # Feature Values Normalization
        if normalize == True:
            per_feature_values_normalized = []
            for feature_values in per_feature_values:
                # Normalizing to Standard Normal 
                feature_values = (sp.stats.rankdata(feature_values, method = 'ordinal') - np.mean(per_era_values)) / len(feature_values)
                # Standard Normal Percentile
                feature_values = sp.stats.norm.ppf(feature_values)
                per_feature_values_normalized.append(feature_values)
            per_era_values = np.array(per_feature_values_normalized).T
        exposures = data_era[features_to_neutralize].values
        per_era_values -= exposures.dot(np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(per_era_values.astype(np.float32)))
        per_era_values /= per_era_values.std(ddof = 0)
        pred_neutralized.append(per_era_values)
    pred_neutralized = np.concatenate(pred_neutralized).T[0]
    return pred_neutralized

def neutralization_series(series, by):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)
    exposures = np.hstack((exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))
    correction = exposures.dot(np.linalg.lstsq(exposures, scores, rcond = None)[0])
    scores_corrected = scores - correction
    series_neutralized = pd.Series(scores_corrected.ravel(), index = series.index)
    return series_neutralized

def validation_metrics(data, index, pred, pred_col = "pred", target_col = "target", include_correlations = False):
    validation_stats = pd.DataFrame()
    # data = data[data["era"].isin(index)]
    data = data.loc[index, :]
    data[pred_col] = pred.copy()
    data[pred_col] = data[pred_col].rank(pct = True)
    # Prediction Correlations with Target per Era
    validation_corr = data.groupby("era").apply(lambda d: unif(d[pred_col]).corr(d[target_col]))
    # Validation Stats: Mean, Standard Deviation and Sharpe Ratio
    if include_correlations == True:
        mean = validation_corr.mean()
        std = validation_corr.std(ddof = 0)
        sharpe_ratio = mean / std
        # Add Validation Stats
        validation_stats.loc["corr_mean", pred_col] = mean
        validation_stats.loc["corr_std", pred_col] = std
        validation_stats.loc["corr_sharpe", pred_col] = sharpe_ratio
    validation_stats = validation_stats.transpose()
    return validation_stats

def grid_search_results(df, run, grid, test_score):
    df.loc[f"Run{run}", "Params"] = [grid]
    df.loc[f"Run{run}", "Test_Score"] = test_score
    return df

def grid_search_best_params(df):
    best_params_row = df[df["Test_Score"] == df["Test_Score"].max()]
    best_params = best_params_row["Params"]
    return best_params

def cross_validation(data_train, model, params, k = 3, neutralizing = False):
    folds_train, folds_test = train_test_folds(data_train, k, embargo = 12)
    features = [col for col in data_train if col.startswith("feature")]
    riskiest_features = risk_features(data_train, n = 50)
    # K-Fold Cross Validation
    run = 1
    fold_test_scores = []
    for fold_train, fold_test in zip(folds_train, folds_test):
        print(f"Run # {run}/{k}")
        # Train-Test-Split
        fold_train_index = data_train["era"].isin(fold_train)
        fold_test_index = data_train["era"].isin(fold_test)
        X_train = data_train.loc[fold_train_index, features]
        X_test = data_train.loc[fold_test_index, features]
        y_train = data_train.loc[fold_train_index, ["target"]]
        fold_train_int = [int(i[3:]) for i in fold_train]
        fold_test_int = [int(i[3:]) for i in fold_test]
        fold_train_min, fold_train_max = int(np.min(fold_train_int)), int(np.max(fold_train_int))
        fold_test_min, fold_test_max = int(np.min(fold_test_int)), int(np.max(fold_test_int))
        print(f"Train Fold Eras: {fold_train_min}-{fold_train_max}")
        print(f"Test Fold Eras: {fold_test_min}-{fold_test_max}")
        # Model Fit
        model.set_params(**params)  
        model.fit(X_train, y_train) 
        # Model Prediction
        pred = model.predict(X_test)
        if neutralizing == True:
            pred = neutralization(data = data_train.loc[fold_test_index, :], columns_to_neutralize = ["pred"], features_to_neutralize = riskiest_features, pred = pred, normalize = True)
        # Model Evaluation
        fold_test_stats = validation_metrics(data_train, fold_test_index, pred, pred_col = "pred", target_col = "target", include_correlations = True)
        fold_test_score = fold_test_stats["corr_sharpe"]
        fold_test_scores.append(fold_test_score)
        print(f"Test Fold Stats:\n {fold_test_stats}")
        print(f"Test Fold Score: {fold_test_score[0]}\n")
        # Update Run 
        run += 1
    fold_test_scores_mean = np.mean(np.array(fold_test_scores))
    print(f"Aggregated Test Score: {fold_test_scores_mean}\n")
    return fold_test_scores_mean

def grid_search_cross_validation(data_train, model, grid, neutralizing = False):
    print("---------- Grid Search K-Fold Cross Validation ----------")
    # Grid Search K-Fold Cross Validation
    keys = list(grid.keys())
    results = pd.DataFrame(columns = ["Params", "Test_Score"])
    run = 1
    if len(grid) == 1:
        for param_1 in grid[keys[0]]:
            param_grid = {keys[0]: param_1}
            print(f"Params: {param_grid}\n")
            fold_test_scores_mean = cross_validation(data_train, model, params = param_grid, k = 3, neutralizing = neutralizing)
            results = grid_search_results(results, run, param_grid, fold_test_scores_mean)
            run += 1
    elif len(grid) == 2:
        for param_1, param_2 in [(param_1, param_2) for param_1 in grid[keys[0]] for param_2 in grid[keys[1]]]:
            param_grid = {keys[0]: param_1, keys[1]: param_2}
            print(f"Params: {param_grid}\n")
            fold_test_scores_mean = cross_validation(data_train, model, params = param_grid, k = 3, neutralizing = neutralizing)
            results = grid_search_results(results, run, param_grid, fold_test_scores_mean)
            run += 1
    elif len(grid) == 3:
        for param_1, param_2, param_3 in [(param_1, param_2, param_3) for param_1 in grid[keys[0]] for param_2 in grid[keys[1]] for param_3 in grid[keys[2]]]:
            param_grid = {keys[0]: param_1, keys[1]: param_2, keys[2]: param_3}
            print(f"Params: {param_grid}\n")
            fold_test_scores_mean = cross_validation(data_train, model, params = param_grid, k = 3, neutralizing = neutralizing)
            results = grid_search_results(results, run, param_grid, fold_test_scores_mean)
            run += 1
    elif len(grid) == 4:
        for param_1, param_2, param_3, param_4 in [(param_1, param_2, param_3, param_4) for param_1 in grid[keys[0]] for param_2 in grid[keys[1]] for param_3 in grid[keys[2]] for param_4 in grid[keys[3]]]:
            param_grid = {keys[0]: param_1, keys[1]: param_2, keys[2]: param_3, keys[3]: param_4}
            print(f"Params: {param_grid}\n")
            fold_test_scores_mean = cross_validation(data_train, model, params = param_grid, k = 3, neutralizing = neutralizing)
            results = grid_search_results(results, run, param_grid, fold_test_scores_mean)
            run += 1
    best_params = grid_search_best_params(results)
    print(f"Best Params: {best_params[0]}")
    return best_params[0]
