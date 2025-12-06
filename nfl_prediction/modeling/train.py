from pathlib import Path
import pickle

import pandas as pd
from loguru import logger
import typer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import numpy as np

from nfl_prediction.config import MODELS_DIR, MATCHUPS_DIR
from nfl_prediction.modeling.tune_xgb import tune_xgb_hyperparams

app = typer.Typer()

VEGAS_SCALE_MAP = {
    "home_moneyline": 0.0,
    "away_moneyline": 0.0,
    "home_implied_prob": 0.15,
    "vegas_spread": 0.1,
}

def scale_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent scaling to Vegas-related columns.
    """
    for col, factor in VEGAS_SCALE_MAP.items():
        if col in df.columns:
            df[col] = df[col] * factor
        else:
            logger.warning(f"Vegas column '{col}' not found in dataframe during scaling.")
    return df

def make_rolling_splits(df: pd.DataFrame, start_season: int = 2022) -> list[dict]:
    """
    Rolling-origin train/val/test splits.
    
    For each season szn:
        - Train on seasons < szn
        - Val on szn weeks 1-5
        - Test on szn weeks 6+
    """
    
    seasons = sorted(df["season"].unique())
    
    splits = []
    
    for szn in seasons:
        if szn == start_season:
            continue
        
        train_split = df["season"] < szn
        val_split = (df["season"] == szn) & (df["week"].between(1, 5))
        test_split = (df["season"] == szn) & (df["week"] >= 6)
        
        splits.append({
            "test_season": szn,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
        })
        
    return splits

"""
def make_splits(df: pd.DataFrame):
    '''
    Time-based train/val/test splits
    
    — Train: 2022 - 2025 (week 5)
    — Validate: 2025 weeks 6-8
    — Test: 2025 weeks 9-12
    '''
    season_col = "season"
    week_col = "week"
    
    train_set = (df[season_col] < 2025) | ((df[season_col] == 2025) & df[week_col] <= 5)
    val_set = (df[season_col] == 2025) & (df[week_col].between(6, 8))
    test_set = (df[season_col] == 2025) & (df[week_col] >= 9)
    
    return train_set, val_set, test_set
"""
def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Return select features to be used in model training.
    """
    selected_features = [
        # Vegas (scaled)
        "vegas_spread",
        "home_moneyline",
        "away_moneyline",
        "home_implied_prob",

        # Elo ratings
        "home_elo_pre",
        "away_elo_pre",
        "diff_elo_pre",

        # SHAP
        # "opp_puntaverage",
        "passingattempts",
        "rushingyardsperattempt",
        # "opp_opponentpassingyardspercompletion",
        "quarterbackhits",
        # "puntyards",
        "opp_rolling_win_rate_5",
        "timessackedyards",
        # "opp_opponentassistedtackles",
        "opponentpenaltyyards",
        # "opponentrushingyards",
        # "opponenttimeofpossessionseconds",
        # "opp_solotackles",
        # "opp_opponenttimessackedyards",

        # Rolling stability features
        "rolling_yards_total_3",
        "rolling_points_for_3",
        "rolling_points_against_3",
        "opp_rolling_yards_total_3",
        "opp_rolling_points_for_3",
        "opp_rolling_points_against_3",
        
        # Rolling EPA metrics (from nflfastR)
        "off_epa_per_play_rolling_3",
        "off_dropback_epa_rolling_3",
        "def_epa_per_play_rolling_3",
        "def_dropback_epa_against_rolling_3",
        "def_success_rate_rolling_3",
        "def_rush_epa_against_rolling_3",
        "off_success_rate_rolling_3",
        "off_rush_epa_rolling_3",
        "epa_off_diff_rolling_3",
        "epa_def_diff_rolling_3",
        "off_epa_per_play_rolling_5",
        "def_epa_per_play_rolling_5",
    ]
    
    final_cols = [c for c in selected_features if c in df.columns]
    
    missing = set(selected_features) - set(final_cols)
    if missing:
        print(f"[WARNING] Missing expected features: {missing}")
    
    return final_cols

@app.command()
def main(features_path: Path = MATCHUPS_DIR / "matchups_all_seasons.csv", 
         model_path: Path = MODELS_DIR / "xgb_point_diff.pkl"):
    logger.info(f"Loading data from {features_path}")
    df = pd.read_csv(features_path)
    
    target_col = "point_diff"
    
    df = df.dropna(subset=[target_col])
    
    df = scale_vegas_features(df)

    feature_cols = select_features(df, target_col)
    logger.info(f"Using {len(feature_cols)} features")
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df[target_col]
    
    df["next_game_win"] = (df[target_col] > 0).astype(int)
    win = df["next_game_win"]
    
    splits = make_rolling_splits(df)
    
    all_metrics = []
    
    for split in splits:
        test_season = split["test_season"]
        train_set = split["train_split"]
        val_set = split["val_split"]
        test_set = split["test_split"]
        
        X_train, y_train = X[train_set], y[train_set]
        X_val, y_val = X[val_set], y[val_set]
        X_test, y_test = X[test_set], y[test_set]
        
        win_train = win[train_set]
        win_val = win[val_set]
        win_test = win[test_set]
        
        logger.info(
            f"\n==== Rolling split: Test season {test_season} ====\n"
            f"Train: {len(X_train)} rows | Val: {len(X_val)} rows | Test: {len(X_test)} rows"
        )
    
        # best_params = tune_xgb_hyperparams(df, feature_cols, target_col)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            eval_metric="rmse",
            random_state=34,
            n_jobs=1,
            # **best_params,
            max_depth=1,
            min_child_weight=14,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_lambda=3.0,
            reg_alpha=0.0,
        )

        logger.info(f"[Season {test_season}] Training XGBoost model...")
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
            
            logger.info(
                f"[Season {test_season}] {name} "
                f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f} | WIN ACCURACY={win_acc:.3f}"
            )
            return mae, rmse, r2, win_acc
    
        report_split("Train", X_train, y_train, win_train)
        report_split("Val", X_val, y_val, win_val)
        mae_test, rmse_test, r2_test, winacc_test = report_split("Test",  X_test,  y_test,  win_test)
        
        preds_test = model.predict(X_test)
        spread_mae = np.mean(np.abs(preds_test - y_test))
        spread_bias = np.mean(preds_test - y_test)
        logger.info(f"[Season {test_season}] Spread Bias (positive = overpredicting home): {spread_bias:.3f}\n")
        
        all_metrics.append({
            "test_season": test_season,
            "test_MAE": mae_test,
            "test_RMSE": rmse_test,
            "test_R2": r2_test,
            "test_WIN_ACC": winacc_test,
            "test_Spread_MAE": spread_mae,
            "test_Spread_Bias": spread_bias,
        })
    

    final_train_mask = df["point_diff"].notna()
    X_final = X[final_train_mask]
    y_final = y[final_train_mask]
    
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        eval_metric="rmse",
        random_state=34,
        n_jobs=1,
        # **best_params,
        max_depth=1,
        min_child_weight=14,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_lambda=3.0,
        reg_alpha=0.0,
    )
        
    logger.info(f"Training FINAL model on all completed games...")
    final_model.fit(X_final, y_final, verbose=False)
        
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump({"model": final_model, "features": feature_cols}, file)
    
    logger.success(f"Saved FINAL trained model to {model_path}")

if __name__ == "__main__":
    app()