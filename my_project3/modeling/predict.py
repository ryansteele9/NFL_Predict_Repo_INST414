from pathlib import Path
import pickle
from typing import Optional, List
import os

import numpy as np
import pandas as pd
from loguru import logger
import typer
import requests

from my_project3.data.build_matchup_data import add_matchup_strength_features
from my_project3.config import MODELS_DIR, MATCHUPS_DATA_DIR, FEATURES_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Predict NFL game outcomes using trained XGBoost model.")

SCHEDULES_DIR = RAW_DATA_DIR / "schedules"
SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)

SPORTSDATA_API_KEY = os.environ.get("SPORTSDATAIO_API_KEY")


def default_model_path() -> Path:
    return MODELS_DIR / "xgb_point_diff.pkl"

def load_model_and_features(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError
    
    logger.info(f"Loading Model from {model_path}")
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    
    model = data["model"]
    if "feature_cols" in data:
        feature_cols = data["feature_cols"]
    elif "features" in data:
        feature_cols = data["features"]
    else:
        raise ValueError(
            "Model pickle must contain 'feature_cols' or 'features' "
            "listing the feature column names."
        )
    logger.info(f"Loaded model with {len(feature_cols)} features.")
    return model, feature_cols

def fetch_schedules(season: int, week:int) -> pd.DataFrame:
    """
    Call Sportradar weekly schedule API.
    
    Args:
        season (int): season year to get schedule from
        week (int): week # to get schedule from
        
    Returns Data Frame with information about future matchups
    """
    
    season_param = f"{season}REG"
    
    url = (f"https://api.sportsdata.io/api/nfl/odds/json/ScoresByWeek/{season_param}/{week}?key={SPORTSDATA_API_KEY}")
    
    logger.info(f"Fetching ScoresByWeek from SportsDataIO: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    if not data:
        raise ValueError(f"No games returned for season={season_param}, week={week}")
    
    df = pd.DataFrame(data)
    
    rename_map = {
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "GameKey": "game_id",
        "ScoreID": "game_id",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    required = {"home_team", "away_team", "game_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ScoresByWeek missing required columns: {missing}")
    
    df["season"] = season
    df["week"] = week

    return df[["season", "week", "game_id", "home_team", "away_team"]]

def load_schedule(season: int, week: int, season_type: str = "REG") -> pd.DataFrame:
    csv_path = SCHEDULES_DIR / f"scoresbyweek_{season}{season_type}_{week:02d}.csv"
    if csv_path.exists():
        logger.info(f"Loading schedule from cache: {csv_path}")
        return pd.read_csv(csv_path)

    df = fetch_schedules(season, week)
    df.to_csv(csv_path, index=False)
    logger.info(f"Cached schedule to {csv_path}")
    return df

def team_features_path(season: int, team: str) -> Path:
    season_dir = FEATURES_DATA_DIR / str(season)
    return season_dir / f"{team}_{season}_features.csv"

def get_features_for_future_week(season: int, future_week: int, team: str) -> pd.Series:
    path = team_features_path(season, team)
    if not path.exists():
        raise FileNotFoundError(f"Feature file for team {team} not found at {path}")
    
    df = pd.read_csv(path)
    if "week" not in df.columns:
        raise ValueError(f"'week' column missing in {path}")
    
    df_here = df.loc[df["week"] == future_week]
    if df_here.empty:
        max_week = df["week"].max()
        logger.warning(
            f"No features for {team} week {future_week} in {path}; "
            f"falling back to week {max_week}"
        )
        df_here = df.loc[df["week"] == max_week]
    
    return df_here.sort_values("week").iloc[-1]

def build_future_matchups(season: int, week: int, season_type: str = "REG") -> pd.DataFrame:
    """
    For each given future week, build a single-row matchup using data from dummy rows
    """
    schedule_df = load_schedule(season, week, season_type)
    
    rows = []
    logger.info(f"Building future matchups for {len(schedule_df)} games...")
    
    for _, game in schedule_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        home_feat = get_features_for_future_week(season, week, home)
        away_feat = get_features_for_future_week(season, week, away)
        
        row = {
            "season": season,
            "week": week,
            "game_id": game["game_id"],
            "home_team": home,
            "away_team": away,
        }
        
        for col, val in home_feat.items():
            row[col] = val
        
        for col, val in away_feat.items():
            row[f"opp_{col}"] = val
        
        rows.append(row)
        
    df_matchups = pd.DataFrame(rows)
    df_matchups = add_matchup_strength_features(df_matchups)
    logger.info(f"Constructed {len(df_matchups)} future matchup rows.")
    return df_matchups

def add_predictions(df_games: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df_games.columns]
    if missing:
        raise ValueError(f"Future matchup data missing feature columns: {missing}")
    
    X = df_games[feature_cols].to_numpy()
    preds = model.predict(X)
    
    df_games["pred_point_diff"] = preds
    df_games["pred_winner"] = np.where(preds > 0, "home", "away")
    
    spread = np.abs(preds)
    win_prob = 0.5 + 0.4 * np.tanh(spread / 10.0)
    df_games["pred_home_win_prob"] = np.where(preds > 0, win_prob, 1.0 - win_prob)
    
    return df_games

def print_pred_results(df_pred: pd.DataFrame) -> None:
    cols = [col for col in ["season", "week", "game_id", "home_team", "away_team"] if col in df_pred.columns]
    cols += ["pred_point_diff", "pred_winner", "pred_home_win_prob"]
    
    output = df_pred[cols].copy()
    output["pred_point_diff"] = output["pred_point_diff"].round(2)
    output["pred_home_win_prob"] = output["pred_home_win_prob"].round(3)
    print()
    print(output.to_string(index=False))
    print()

@app.command()
def main(
    season: int = typer.Option(..., help="Season year, e.g. 2025"), 
    week: int = typer.Option(..., help="Future week number, e.g. 13"), 
    season_type: str = typer.Option(
        "REG", help="Season type: REG, PRE, or POST"
    ),
    model_path: Optional[Path] = typer.Option(
        None, help="Optional path to model pickle (defaults to MODELS_DIR)."
    ),
    home_team: Optional[str] = typer.Option(
        None, help="Optional filter: only this home team code (e.g. BAL)."
    ),
    away_team: Optional[str] = typer.Option(
        None, help="Optional filter: only this away team code (e.g. NYJ)."
    ),
    game_id: Optional[str] = typer.Option(
        None, help="Optional filter: only this specific game_id/GameKey."
    ),
    save_matchups: bool = typer.Option(
        False,
        help="If set, save synthetic future matchup rows to data/matchups/.",
    ),
):
    """
    Predict point differential for all games in a scheduled future week.
    """
    model_path = model_path or default_model_path()
    model, feature_cols = load_model_and_features(model_path)
    
    logger.info(f"Predicting season={season}{season_type}, week={week}")
    df_matchups = build_future_matchups(season, week, season_type=season_type)
    
    # optional filters
    mask = pd.Series(True, index=df_matchups.index)
    if home_team:
        mask &= df_matchups["home_team"] == home_team
    if away_team:
        mask &= df_matchups["away_team"] == away_team
    if game_id:
        mask &= df_matchups["game_id"] == game_id
    
    df_matchups = df_matchups.loc[mask].copy()
    if df_matchups.empty:
        raise ValueError("No games match the given filters.")
    
    if save_matchups:
        out_path = MATCHUPS_DATA_DIR / f"future_matchups_{season}{season_type}_week{week:02d}.csv"
        df_matchups.to_csv(out_path, index=False)
        logger.info(f"Saved future matchups to {out_path}")
    
    df_pred = add_predictions(df_matchups, model, feature_cols)
    print_pred_results(df_pred)
    
    logger.success("Prediction Complete.")


if __name__ == "__main__":
    app()
