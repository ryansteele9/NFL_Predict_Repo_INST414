import pandas as pd
from pathlib import Path
from my_project3.config import FEATURES_DATA_DIR, MATCHUPS_DATA_DIR

FEATURE_DIR = FEATURES_DATA_DIR
OUTPUT_DIR = MATCHUPS_DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_matchups_for_season(season: str):
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

    out_path = OUTPUT_DIR / f"matchups_{season}.csv"
    matchups.to_csv(out_path, index=False)
    print(f"Saved {len(matchups)} games to {out_path}")

def main():
    for season in ["2022", "2023", "2024", "2025"]:
        build_matchups_for_season(season)

if __name__ == "__main__":
    main()