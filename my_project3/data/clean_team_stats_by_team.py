import pandas as pd
from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/teams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_cleaned_season(season: str):
    """
    Split clean_team_stats_{season}.csv into 32 team-specific CSVs.
    Each CSV contains weekly rows for exactly one team in that season.
    """
    file_path = INPUT_DIR / f"clean_team_stats_{season}.csv"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\nSplitting {file_path} by team...")

    df = pd.read_csv(file_path)

    season_dir = OUTPUT_DIR / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    for team in df["team"].unique():
        team_df = (
            df[df["team"] == team].sort_values(by="week").reset_index(drop=True)
        )

        safe_name = team.replace(" ", "_")
        out_path = season_dir / f"{safe_name}_{season}.csv"
        team_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(team_df)} rows)")

def main():
    seasons = ["2022", "2023", "2024", "2025"]
    for season in seasons:
        split_cleaned_season(season)

if __name__ == "__main__":
    main()
