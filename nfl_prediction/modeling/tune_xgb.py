"""
Helper function that tunes hyperparameters for model. Uses grid-search to train
model on different sets of parameters. Returns parameters that give best mean
absolute error. Used to get parameters for model in train.py.
"""
import pandas as pd
from loguru import logger

from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np



def tune_xgb_hyperparams(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict:
    """
    Brute-force tune a small XGBoost hyperparameter grid using rolling season splits.
    Train and validate models with different parameters. Evaluate parameters by 
    lowest MAE, parameters resulting in lowest MAE are used for final model.
    """

    seasons = sorted(df["season"].unique())
    
    candidate_test_seasons = [s for s in seasons if s >= 2023]

    logger.info(f"Hyperparameter tuning across test seasons: {candidate_test_seasons}")

    param_grid = [
        {"max_depth": 1, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 2.0, "reg_alpha": 0.0},
        {"max_depth": 1, "min_child_weight": 14, "subsample": 0.9, "colsample_bytree": 0.6, "reg_lambda": 3.0, "reg_alpha": 0.0},

        {"max_depth": 2, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 2.0, "reg_alpha": 0.2},
        {"max_depth": 2, "min_child_weight": 14, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 3.0, "reg_alpha": 0.3},

        {"max_depth": 2, "min_child_weight": 8,  "subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 2.0, "reg_alpha": 0.5},
        {"max_depth": 3, "min_child_weight": 12, "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 3.0, "reg_alpha": 0.5},
    ]


    best_params: dict | None = None
    best_score = np.inf

    X_full = df[feature_cols]
    y_full = df[target_col]

    for i, params in enumerate(param_grid, start=1):
        logger.info(f"Testing param set {i}/{len(param_grid)}: {params}")
        fold_maes: list[float] = []

        for test_season in candidate_test_seasons:
            train_mask = df["season"] < test_season
            val_mask   = df["season"] == test_season

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_train = X_full[train_mask]
            y_train = y_full[train_mask]
            X_val   = X_full[val_mask]
            y_val   = y_full[val_mask]

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=600,
                learning_rate=0.05,
                eval_metric="rmse",
                random_state=34,
                n_jobs=1,
                **params,
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            fold_maes.append(mae)

        if not fold_maes:
            continue

        avg_mae = float(np.mean(fold_maes))
        logger.info(f"Param set {i}: avg validation MAE across seasons = {avg_mae:.3f}")

        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params

    if best_params is None:
        raise RuntimeError("Hyperparameter tuning failed: no valid folds/params evaluated.")

    logger.success(f"Best params: {best_params} with avg MAE={best_score:.3f}")
    return best_params