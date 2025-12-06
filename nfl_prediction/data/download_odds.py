"""
Downloads Vegas odds from SportsDataIO API GameOddsByWeek endpoint. Coverts odds
to probabilities.

Returns:
    Raw Vegas odds files, saved to: raw/odds/
    Processed Vegas odds files, saved to: processed/odds
"""
import argparse
import numpy as np
import requests
import pandas as pd

from nfl_prediction.config import ODDS_RAW_DIR, ODDS_PROC_DIR, SPORTSDATAIO_API_KEY

def american_to_prob(odds: float) -> float:
    """
    Convert American odds to probabilities.
    """
    
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)
    
def normalize_prob(p_home: float, p_away: float) -> (float, float):
    """
    Normalize probabilities.
    """
    
    total = p_home + p_away
    
    if total == 0:
        return 0.5, 0.5
    
    return p_home / total, p_away / total

def fetch_odds(season, week):
    """
    Fetch json from API.
    """
    url = f"https://api.sportsdata.io/api/nfl/odds/json/GameOddsByWeek/{season}/{week}?key={SPORTSDATAIO_API_KEY}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
    

def parse_odds_response(game_info, season: int, week: int) -> pd.DataFrame:
    """
    Parse odds API response into DataFrame (matchup-level).
    """
    
    records = []
    
    for game in game_info:
        scoreid = game.get("ScoreId")
        home = game.get("HomeTeamName")
        away = game.get("AwayTeamName")
        
        odds_list = game.get("PregameOdds") or []
        if not odds_list:
            continue
        
        spreads = []
        totals = []
        home_ml = []
        away_ml = []
        
        for odds in odds_list:
            if odds.get("HomePointSpread") is not None:
                spreads.append(float(odds["HomePointSpread"]))
            if odds.get("OverUnder") is not None:
                totals.append(float(odds["OverUnder"]))
            if odds.get("HomeMoneyLine") is not None:
                home_ml.append(float(odds["HomeMoneyLine"]))
            if odds.get("AwayMoneyLine") is not None:
                away_ml.append(float(odds["AwayMoneyLine"]))
        
        vegas_spread = np.mean(spreads) if spreads else None
        vegas_total = np.mean(totals) if totals else None
        home_ml_avg = np.mean(home_ml) if home_ml else None
        away_ml_avg = np.mean(away_ml) if away_ml else None
        
        prob_home_raw = american_to_prob(home_ml_avg) if home_ml_avg is not None else None
        prob_away_raw = american_to_prob(away_ml_avg) if away_ml_avg is not None else None
        
        home_prob = None
        if prob_home_raw is not None and prob_away_raw is not None:
            home_prob, _ = normalize_prob(prob_home_raw, prob_away_raw)
        
        records.append({
            "season": season,
            "week": week,
            "game_id": scoreid,
            "home_team": home,
            "away_team": away,
            "vegas_spread": vegas_spread,
            "vegas_total": vegas_total,
            "home_moneyline": home_ml_avg,
            "away_moneyline": away_ml_avg,
            "home_implied_prob": home_prob,
        })
        
    return pd.DataFrame(records)

def save_week(df: pd.DataFrame, season: int, week: int):
    """
    Saves input odds dataset in correct folder.
    """
    ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    ODDS_PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_path = ODDS_RAW_DIR / f"odds_{season}_week{week:02d}.csv"
    clean_path = ODDS_PROC_DIR / f"odds_{season}_week{week:02d}.csv"
    
    df.to_csv(raw_path, index=False)
    df.to_csv(clean_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Download a single season")
    parser.add_argument("--week", type=int, help="Optional single week for that season")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()
    
    if args.season is not None:
        seasons = [args.season]
    else:
        seasons = list(range(args.start_season, args.end_season + 1))
        
    for season in seasons:
        if args.week is not None:
            weeks = [args.week]
        else:
            weeks = list(range(1, 19))
        
        for week in weeks:
            print(f"Fetching odds for {season} Week {week}")
            data = fetch_odds(season, week)
            df = parse_odds_response(data, season, week)
            print("Rows:", len(df))
            save_week(df, season, week)

if __name__ == "__main__":
    main()