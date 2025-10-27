import pandas as pd
from pathlib import Path

MATCHUP_DIR = Path("my_project3/data/matchups")
OUTPUT_PATH = MATCHUP_DIR / "matchups_all_seasons.csv"

def combine_matchups():
    all_dfs = []
    for season in ["2022", "2023", "2024", "2025"]:
        file_path = MATCHUP_DIR / f"matchups_{season}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["season"] = int(season)
            all_dfs.append(df)
            print(f"Loaded {file_path} with {len(df)} rows")
        else:
            print(f"Missing file: {file_path}, skipping...")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nCombined dataset saved to: {OUTPUT_PATH}")
        print(f"Total rows: {len(full_df)}")
    else:
        print("No matchup files found!")

if __name__ == "__main__":
    combine_matchups()
