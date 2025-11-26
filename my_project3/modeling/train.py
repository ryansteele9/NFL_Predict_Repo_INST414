from pathlib import Path
import pickle

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from my_project3.config import MODELS_DIR, MATCHUPS_DATA_DIR

app = typer.Typer()

def make_splits(df: pd.DataFrame):
    """
    Time-based train/val/test splits
    
    — Train: 2022 - 2025 (week 5)
    — Validate: 2025 weeks 6-10
    — Test: 2025 weeks 11-12
    """
    season_col = "season"
    week_col = "week"
    
    train_set = (df[season_col] < 2025) | ((df[season_col] == 2025) & df[week_col] <= 5)
    val_set = (df[season_col] == 2025) & (df[week_col].between(6, 8))
    test_set = (df[season_col] == 2025) & (df[week_col] >= 9)
    
    return train_set, val_set, test_set

def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Use all numeric columns except IDs, metadata, target (next_point_diff)
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
    }
    
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    return feature_cols


@app.command()
def main(features_path: Path = MATCHUPS_DATA_DIR / "matchups_all_seasons.csv", 
         model_path: Path = MODELS_DIR / "xgb_point_diff.pkl"):
    logger.info(f"Loading data from {features_path}")
    df = pd.read_csv(features_path)
    
    target_col = "target_next_point_diff"
    
    before_drop = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"Dropped {before_drop - len(df)} rows with missing target column")
    
    feature_cols = select_features(df, target_col)
    logger.info(f"Using {len(feature_cols)} features")
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df[target_col]
    
    train_set, val_set, test_set = make_splits(df)
    
    X_train, y_train = X[train_set], y[train_set]
    X_val, y_val = X[val_set], y[val_set]
    X_test, y_test = X[test_set], y[test_set]
    
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
    
    def report_split(name: str, X_split, y_split):
        predictions = model.predict(X_split)
        mae = mean_absolute_error(y_split, predictions)
        rmse = mean_squared_error(y_split, predictions, squared=False)
        r2 = r2_score(y_split, predictions)
        logger.info(f"{name} MAE={mae} | RMSE={rmse} | R2={r2}")
    
    report_split("Train", X_train, y_train)
    report_split("Val", X_val, y_val)
    if len(X_test) > 0:
        report_split("Test", X_test, y_test)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump({"model": model, "features": feature_cols}, file)
    
    logger.success(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    app()