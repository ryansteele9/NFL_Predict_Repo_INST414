import pandas as pd
from pathlib import Path
from my_project3.config import MATCHUPS_DATA_DIR
from my_project3.data.team_ratings import add_elo_features


MATCHUP_DIR = MATCHUPS_DATA_DIR
OUTPUT_PATH = MATCHUP_DIR / "matchups_all_seasons.csv"

def combine_matchups():
    all_dfs = []
    for season in ["2022", "2023", "2024", "2025"]:
        file_path = MATCHUP_DIR / f"matchups_{season}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["season"] = int(season)
            
            if "point_diff" in df.columns:
                before = len(df)
                df = df[df["point_diff"].notna()].copy()
                after = len(df)
                print(f"""Loaded {file_path} with {before} rows.
                      {before - after} rows with NaN point_diff dropped.
                      """)
            else:
                print(
                    f"WARNING: 'point_diff' not in {file_path} columns; "
                    "no filtering of dummy rows applied."
                )
            all_dfs.append(df)
        else:
            print(f"Missing file: {file_path}, skipping...")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        if "gamekey" not in full_df.columns:
            full_df = full_df.sort_values(["season", "week"])
        else:
            full_df = full_df.sort_values(["season", "week", "gamekey"])
        
        full_df = add_elo_features(full_df)
        
        full_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nCombined dataset saved to: {OUTPUT_PATH}")
        print(f"Total rows: {len(full_df)}")
    else:
        print("No matchup files found!")

if __name__ == "__main__":
    combine_matchups()
