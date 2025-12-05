import pandas as pd
from pathlib import Path
import numpy as np
from my_project3.config import FEATURES_DATA_DIR, MATCHUPS_DATA_DIR

FEATURE_DIR = FEATURES_DATA_DIR
OUTPUT_DIR = MATCHUPS_DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRENGTH_BASE_COLS = [
    "rolling_point_diff_3",
    "rolling_win_rate_5",
    "rolling_yards_total_3",
    "rolling_yards_allowed_3",
    "cumulative_points_for",
    "cumulative_points_against",
    "cumulative_wins",
]

def add_matchup_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates strength features from existing rolling averages and cumulative stats.
    """
    for col in STRENGTH_BASE_COLS:
        home_col = col
        opp_col = f"opp_{col}"
        
        if home_col in df.columns and opp_col in df.columns:
            diff_name = f"strength_diff_{col}"
            sum_name = f"strength_sum_{col}"

            df[diff_name] = df[home_col] - df[opp_col]
            df[sum_name] = df[home_col] + df[opp_col]
    
    if "off_epa_per_play_rolling_3" and "opp_off_epa_per_play_rolling_3" in df.columns:
        df["epa_off_diff_rolling_3"] = (
            df["off_epa_per_play_rolling_3"] 
            - df["opp_off_epa_per_play_rolling_3"]
        )
    if "def_epa_per_play_rolling_3" and "opp_def_epa_per_play_rolling_3" in df.columns:
        df["epa_def_diff_rolling_3"] = (
            df["def_epa_per_play_rolling_3"] 
            - df["opp_def_epa_per_play_rolling_3"]
        )
    
    return df

def build_matchups_for_season(season: str):
    """
    Creates all weekly matchups for a given season.
    """
    print(f"\nBuilding matchup data for season {season}...")

    season_dir = FEATURE_DIR / season
    if not season_dir.exists():
        print(f"No feature data found for {season}")
        return

    all_teams = pd.concat([pd.read_csv(f) for f in season_dir.glob("*.csv")], ignore_index=True)
    opp_df = all_teams.add_prefix("opp_")

    matchups = all_teams.merge(
        opp_df,
        left_on=["gamekey", "team", "opponent"],
        right_on=["opp_gamekey", "opp_opponent", "opp_team"],
        how="inner"
    )

    if "home_away" in matchups.columns:
        matchups = matchups[matchups["home_away"] == "HOME"]
    
    if "date" in matchups.columns:
        matchups = matchups.sort_values(by=["week", "date"]).reset_index(drop=True)
    else:
        matchups = matchups.sort_values(by="week").reset_index(drop=True)

    matchups = add_matchup_strength_features(matchups)
    
    out_path = OUTPUT_DIR / f"matchups_{season}.csv"
    matchups.to_csv(out_path, index=False)
    print(f"Saved {len(matchups)} games to {out_path}")

def main():
    for season in ["2022", "2023", "2024", "2025"]:
        build_matchups_for_season(season)

if __name__ == "__main__":
    main()