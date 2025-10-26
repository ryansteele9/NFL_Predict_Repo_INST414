import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def split_season_file(season: str):
    """
    Split team_stats_{season}_all_weeks.csv into one file per team.
    """
    file_path = RAW_DIR / f"team_stats_{season}REG_all_weeks.csv"

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return

    df = pd.read_csv(file_path)

    season_folder = PROC_DIR / season
    season_folder.mkdir(parents=True, exist_ok=True)

    teams = df["team"].unique() if "team" in df.columns else df["Team"].unique()

    for team in teams:
        team_name = team.replace(" ", "_")
        team_df = df[df["team"] == team] if "team" in df.columns else df[df["Team"] == team]

        if "week" in team_df.columns:
            team_df = team_df.sort_values(by="week")
        elif "Week" in team_df.columns:
            team_df = team_df.sort_values(by="Week")

        out_path = season_folder / f"{team_name}_{season}.csv"
        team_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

def main():
    seasons = ["2022", "2023", "2024", "2025"]

    for season in seasons:
        print(f"\nProcessing season {season}...")
        split_season_file(season)

if __name__ == "__main__":
    main()
