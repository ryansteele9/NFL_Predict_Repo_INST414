"""
Downloads team-level stats for each season (2022-2025) and week (1-18) from
SportsDataIO API TeamGameStats endpoint. Saves raw JSON files to 
data/raw/sdio_json/. Saves raw CSV files to data/raw/team_stats_csv/. 
Concatenates all weekly CSV files into one file for each season, saved to 
data/raw/team_stats_csv/.
"""
import os
import time
import json
from typing import List
from dotenv import load_dotenv
from pathlib import Path
import requests
import pandas as pd
from nfl_prediction.config import RAW_STATS_JSON, RAW_STATS_CSV

load_dotenv()
API_KEY = os.getenv("SPORTSDATAIO_API_KEY")
BASE = "https://api.sportsdata.io/api/nfl"

RAW_STATS_JSON.mkdir(parents=True, exist_ok=True)
RAW_STATS_CSV.mkdir(parents=True, exist_ok=True)

SLEEP = float(os.getenv("SDIO_SLEEP_SEC", "0.4"))

def get(url: str, params=None) -> requests.Response:
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    return requests.get(url, headers=headers, params=params, timeout=30)

def team_game_stats_week(season: str, week: int) -> pd.DataFrame:
    """
    Download team game stats for a given season and given week.
    
    Args:
        season (str): Season year, ie., '2024REG'
        week (int): week # to download
        
    Returns:
        Pandas Data Frame
    """
    url = f"{BASE}/odds/json/TeamGameStats/{season}/{week}"
    r = get(url)
    if r.status_code == 200:
        data = r.json()
        raw_path = RAW_STATS_JSON / f"sdio_team_stats_{season}_week{week}.json"
        with open(raw_path, "w") as file:
            json.dump(data, file)
            
        df = pd.json_normalize(data)
        
        csv_path = RAW_STATS_CSV / f"team_stats_{season}_week{week}.csv"
        df.to_csv(csv_path, index=False)
        return df
    elif r.status_code in (429, 503):
        time.sleep(2.0)
        return team_game_stats_week(season, week)
    else:
        print(f"Week {week} {season}: {r.status_code}: {r.text}")
        return pd.DataFrame()

def download_season(season: str, weeks: int = 18):
    """
    Download a whole given season.
    
    Args:
        season (str): season to download
        weeks (int): how many weeks of given season to download
    """
    all_rows = []
    for week in range(1, weeks + 1):
        df = team_game_stats_week(season, week)
        if df.empty:
            print(f"{season} Week {week}: no data")
        else:
            if "Season" not in df.columns:
                df["Season"] = season
            if "Week" not in df.columns:
                df["Week"] = week
            all_rows.append(df)
            print(f"{season} Week {week}: {len(df)} rows.")
        time.sleep(SLEEP)
    
    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
    else:
        full = pd.DataFrame()

    output = RAW_STATS_CSV / f"team_stats_{season}_all_weeks.csv"
    full.to_csv(output, index=False)
    print(f"{season} :: Saved season file: {output} ({len(full)} rows)")

def main():
    seasons: List[str] = [
        "2022REG", "2023REG", "2024REG", "2025REG", 
    ]
    for season in seasons:
        print(f"\n====== Downloading {season} ======")
        download_season(season, weeks=18)

if __name__ == "__main__":
    main()