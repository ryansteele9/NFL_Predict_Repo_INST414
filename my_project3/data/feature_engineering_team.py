import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from my_project3.config import TEAMS_DATA_DIR, FEATURES_DATA_DIR 

TEAM_DIR = TEAMS_DATA_DIR
FEATURE_DIR = FEATURES_DATA_DIR 
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def add_future_dummy_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dummy week for upcoming week. Because data is shifted, dummy future week
    will house data from the most recent week. NaNs for columns whose data 
    doesn't exist yet for future dummy week.
    """
    required = {"season", "week", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for dummy creation: {missing}")
    
    groups = []
    for (season, team), g in df.groupby(["season", "team"]):
        g = g.sort_values("week").copy()
        max_week = int(g["week"].max())
        
        dummy = g.iloc[-1].copy()
        dummy["week"] = max_week + 1
        
        unknown_features = {
            "gamekey", "points_for", "points_against", "totalscore", "scorequarter1", 
            "scorequarter2", "scorequarter3", "scorequarter4", "scoreovertime", 
            "opponentscorequarter1", "opponentscorequarter2", "opponentscorequarter3", 
            "opponentscorequarter4", "opponentscoreovertime", "point_diff", "date"
        }
        
        for feature in unknown_features:
            if feature in dummy.index:
                dummy[feature] = np.nan
        
        groups.append(pd.concat([g, pd.DataFrame([dummy])], ignore_index=True))
    
    return pd.concat(groups, ignore_index=True)

def add_team_features(team_file_path):
    """
    Adds rolling and cumulative features to a single team-season file.
    """
    df = pd.read_csv(team_file_path)

    df = df.sort_values(by="week").reset_index(drop=True)
    
    df = add_future_dummy_week(df)

    df["rolling_points_for_3"] = df["points_for"].shift(1).rolling(3).mean()
    df["rolling_points_against_3"] = df["points_against"].shift(1).rolling(3).mean()
    df["rolling_point_diff_3"] = df["point_diff"].shift(1).rolling(3).mean()

    if "offensiveyards" in df.columns:
        df["rolling_yards_total_3"] = df["offensiveyards"].shift(1).rolling(3).mean()
    if "opponentoffensiveyards" in df.columns:
        df["rolling_yards_allowed_3"] = df["opponentoffensiveyards"].shift(1).rolling(3).mean()
    if "turnover_diff" in df.columns:
        df["rolling_turnover_diff_3"] = df["turnover_diff"].shift(1).rolling(3).mean()
    if "third_down_pct" in df.columns:
        df["rolling_third_down_pct_3"] = df["third_down_pct"].shift(1).rolling(3).mean()

    df["rolling_win_rate_5"] = df["win"].shift(1).rolling(5).mean()
    df["cumulative_points_for"] = df["points_for"].shift(1).cumsum()
    df["cumulative_points_against"] = df["points_against"].shift(1).cumsum()
    df["cumulative_wins"] = df["win"].shift(1).cumsum()
    
    do_not_lag = {
        "gamekey", "season", "week", "date", "team", "opponent", "home_away", 
        "stadium", "dayofweek", "teamgameid", "teamid", "opponentid", "win", 
        "points_for", "points_against", "point_diff", "totalscore", 
        "scorequarter1", "scorequarter2", "scorequarter3", "scorequarter4", 
        "scoreovertime", "opponentscorequarter1", "opponentscorequarter2", 
        "opponentscorequarter3", "opponentscorequarter4", "opponentscoreovertime"
    }
    
    already_shifted = {
        "rolling_points_for_3", "rolling_points_against_3", 
        "rolling_point_diff_3", "rolling_yards_total_3", "rolling_yards_allowed_3", 
        "rolling_turnover_diff_3", "rolling_third_down_pct_3", "rolling_win_rate_5", 
        "cumulative_points_for", "cumulative_points_against", "cumulative_wins"
    }
    
    feature_cols = [
        col for col in df.columns 
        if col not in do_not_lag 
        and col not in already_shifted 
        and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    if feature_cols:
        df[feature_cols] = df.groupby("team")[feature_cols].shift(1)
        # df = df.dropna(subset=feature_cols) If need to drop rows without data from lag
        
    round_cols_1 = [
        "rolling_points_for_3", "rolling_points_against_3", 
        "rolling_point_diff_3", "rolling_yards_total_3, rolling_yards_allowed_3"
    ]
    round_cols_3 = [
        "rolling_turnover_diff_3", "rolling_win_rate_5", "third_down_pct", 
        "rolling_third_down_pct_3"
    ]

    for col in round_cols_1:
        if col in df.columns:
            df[col] = df[col].round(1)

    for col in round_cols_3:
        if col in df.columns:
            df[col] = df[col].round(3)
    return df

def process_season(season: str):
    """
    Process all teams in a given season.
    """
    season_team_dir = TEAM_DIR / season
    season_output_dir = FEATURE_DIR / season
    season_output_dir.mkdir(parents=True, exist_ok=True)

    for team_file in season_team_dir.glob("*.csv"):
        team_name = team_file.stem.split("_")[0]
        print(f"Processing {team_name} ({season})")

        df_features = add_team_features(team_file)

        out_path = season_output_dir / f"{team_name}_{season}_features.csv"
        df_features.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")

def main():
    seasons = ["2022", "2023", "2024", "2025"]
    for season in seasons:
        print(f"\nBuilding features for season {season}...")
        process_season(season)

if __name__ == "__main__":
    main()