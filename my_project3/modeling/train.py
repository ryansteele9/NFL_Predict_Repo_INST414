from pathlib import Path
import pickle

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score
import xgboost as xgb
import numpy as np

from my_project3.config import MODELS_DIR, MATCHUPS_DATA_DIR

app = typer.Typer()

def make_splits(df: pd.DataFrame):
    """
    Time-based train/val/test splits
    
    — Train: 2022 - 2025 (week 5)
    — Validate: 2025 weeks 6-8
    — Test: 2025 weeks 9-12
    """
    season_col = "season"
    week_col = "week"
    
    train_set = (df[season_col] < 2025) | ((df[season_col] == 2025) & df[week_col] <= 5)
    val_set = (df[season_col] == 2025) & (df[week_col].between(6, 8))
    test_set = (df[season_col] == 2025) & (df[week_col] >= 9)
    
    return train_set, val_set, test_set

def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Use all numeric columns except IDs, metadata, target (point_diff)
    """
    drop_cols = {
        target_col,
        "gamekey",
        "date",
        "seasontype",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "stadium",
        "teamgameid",
        "dayofweek",
        "teamid",
        "opponentid",
        "scoreid",
        "win",
        "home",
        "point_diff",
        "points_for", 
        "points_against", 
        "timeofpossession",
        "totalscore",
        "opponenttimeofpossession",
        "opp_gamekey",
        "opp_date",
        "opp_seasontype",
        "opp_season",
        "opp_week",
        "opp_team",
        "opp_opponent",
        "opp_home_away",
        "opp_stadium",
        "opp_teamgameid",
        "opp_dayofweek",
        "opp_teamid",
        "opp_opponentid",
        "opp_scoreid",
        "opp_win",
        "opp_home",
        "opp_point_diff",
        "opp_points_for", 
        "opp_points_against", 
        "opp_timeofpossession",
        "opp_totalscore",
        "opp_opponenttimeofpossession",
    }
    
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    return feature_cols


@app.command()
def main(features_path: Path = MATCHUPS_DATA_DIR / "matchups_all_seasons.csv", 
         model_path: Path = MODELS_DIR / "xgb_point_diff.pkl"):
    logger.info(f"Loading data from {features_path}")
    df = pd.read_csv(features_path)
    
    target_col = "point_diff"
    
    before_drop = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"Dropped {before_drop - len(df)} rows with missing target column")
    
    feature_cols = select_features(df, target_col)
    logger.info(f"Using {len(feature_cols)} features")
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df[target_col]
    
    df["next_game_win"] = (df[target_col] > 0).astype(int)
    win = df["next_game_win"]
    
    train_set, val_set, test_set = make_splits(df)
    
    X_train, y_train = X[train_set], y[train_set]
    X_val, y_val = X[val_set], y[val_set]
    X_test, y_test = X[test_set], y[test_set]
    
    win_train = win[train_set]
    win_val   = win[val_set]
    win_test  = win[test_set]
    
    # --- Sanity check: majority baseline on the SAME test split ---
    majority_acc = win_test.mean()   # always predict "win" (1)
    logger.info(f"Test majority baseline (always predict win): {majority_acc:.3f}")
    
    if "home" in df.columns:
        home_pred = df.loc[test_set, "home"].astype(int)
        home_acc = (home_pred == win_test).mean()
        logger.info(f"Always-home baseline: {home_acc:.3f}")


    
    logger.info(f"Train: {len(X_train)} rows\nVal: {len(X_val)} rows\nTest: {len(X_test)} rows")
    
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=34,
        n_jobs=1,
        eval_metric="rmse"
    )

    logger.info("Training XGBoost Model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    def report_split(name: str, X_split, y_split, win_split):
        predictions = model.predict(X_split)
        mae = mean_absolute_error(y_split, predictions)
        rmse = root_mean_squared_error(y_split, predictions)
        r2 = r2_score(y_split, predictions)
        
        pred_win = (predictions > 0).astype(int)
        win_acc = (pred_win == win_split).mean()
        
        logger.info(f"{name} MAE={mae} | RMSE={rmse} | R2={r2} | WIN ACCURACY={win_acc:.3f}")
    
    report_split("Train", X_train, y_train, win_train)
    report_split("Val", X_val, y_val, win_val)
    if len(X_test) > 0:
        report_split("Test", X_test, y_test, win_test)
    
        pred_win_test = (model.predict(X_test) > 0).astype(int)
        cm = confusion_matrix(win_test, pred_win_test)

        logger.info(f"Confusion matrix (Test):\n{cm}")
    
        precision = precision_score(win_test, pred_win_test)
        recall = recall_score(win_test, pred_win_test)
        logger.info(f"Precision: {precision} | Recall: {recall}")
        
            # --- Spread MAE for Test Set ---
        preds_test = model.predict(X_test)
        spread_mae = np.mean(np.abs(preds_test - y_test))
        logger.info(f"Test Spread MAE (avg point error): {spread_mae:.3f}")
        
        spread_bias = np.mean(preds_test - y_test)
        logger.info(f"Spread Bias (positive = overpredicting home): {spread_bias:.3f}")
        
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump({"model": model, "features": feature_cols}, file)
    
    logger.success(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    app()