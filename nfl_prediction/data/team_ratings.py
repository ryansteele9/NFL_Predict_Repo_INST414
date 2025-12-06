"""
Functions that compute elo features given a matchup-level dataset. Functions
implemented in build_full_matchup_data.py.
"""
import pandas as pd
from typing import Dict, Tuple

def add_elo_features(
    df_matchups: pd.DataFrame, 
    base_rating: float = 1500.0, 
    k: float = 20.0, 
    home_field_advantage: float = 2.5
) -> pd.DataFrame:
    """
    Given a matchup-level dataframe, return copy with columns: 
        - home_elo_pre
        - away_elo_pre
        - elo_diff_pre
    Elo ratings updated game-by-game throughout season.
    """
    df = df_matchups.sort_values(["season", "week", "gamekey"]).copy()
    
    ratings: Dict[Tuple[int, str], float] = {}
    home_pre = []
    away_pre = []
    
    for _, row in df.iterrows():
        season = int(row["season"])
        home = row["team"]
        away = row["opponent"]
        margin = float(row["point_diff"])
        
        key_home = (season, home)
        key_away = (season, away)
        
        rating_home = ratings.get(key_home, base_rating)
        rating_away = ratings.get(key_away, base_rating)
        
        home_pre.append(rating_home)
        away_pre.append(rating_away)
        
        rating_diff = (rating_home - rating_away) + home_field_advantage
        expected_home_win = 1.0 / (1.0 + 10 ** (-(rating_diff) / 400.0))
        
        if margin > 0:
            actual_home_win = 1.0
        elif margin < 0:
            actual_home_win = 0.0
        else:
            actual_home_win = 0.5
        
        # margin of victory
        mov_scale = (abs(margin) / 14.0) ** 0.5
        mov_scale = max(0.5, mov_scale)
        
        delta = k * mov_scale * (actual_home_win - expected_home_win)
        
        ratings[key_home] = rating_home + delta
        ratings[key_away] = rating_away - delta
        
    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    df["diff_elo_pre"] = df["home_elo_pre"] - df["away_elo_pre"]
    
    return df

def get_elo_ratings_to_week(
    df_matchups: pd.DataFrame,
    season: int,
    up_to_week: int,
    base_rating: float = 1500.0,
    k: float = 20.0,
    home_field_advantage: float = 2.5,
) -> Dict[str, float]:
    """
    Compute Elo ratings for a given season up to a given week.
    """
    
    df = df_matchups[
        (df_matchups["season"] == season) & (df_matchups["week"] < up_to_week)
    ].copy()
    
    if df.empty:
        return {}
    
    if "gamekey" in df.columns:
        df = df.sort_values(["season", "week", "gamekey"])
    else:
        df = df.sort_values(["season", "week"])
    
    ratings: Dict[str, float] = {}
    
    for _, row in df.iterrows():
        home = row["team"]
        away = row["opponent"]
        margin = float(row["point_diff"])
        
        rating_home = ratings.get(home, base_rating)
        rating_away = ratings.get(away, base_rating)
        
        rating_diff = (rating_home - rating_away) + home_field_advantage
        expected_home_win = 1.0 / (1.0 + 10 ** (-(rating_diff) / 400.0))
        
        if margin > 0:
            actual_home_win = 1.0
        elif margin < 0:
            actual_home_win = 0.0
        else:
            actual_home_win = 0.5
        
        mov_scale = (abs(margin) / 14.0) ** 0.5
        mov_scale = max(0.5, mov_scale)
        
        delta = k * mov_scale * (actual_home_win - expected_home_win)
        
        ratings[home] = rating_home + delta
        ratings[away] = rating_away - delta
        
    return ratings