import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from my_project3.config import DATA_DIR, MODELS_DIR

model_path = MODELS_DIR / "xgb_point_diff.pkl"
with open(model_path, "rb") as file:
    obj = pickle.load(file)
model = obj["model"]
feature_cols = obj["features"]

FEATURES_DIR = DATA_DIR / "features" / "2025"
team_files = list(FEATURES_DIR.glob("*_features.csv"))

teams = {}
for fpath in team_files:
    df = pd.read_csv(fpath)
    team_name = fpath.stem.split("_")[0]
    teams[team_name] = df.set_index("week")

TODAYS_GAMES = [
    ("GB", "DET"),
    ("KC", "DAL"),
    ("CIN", "BAL"),
]

rows = []

for home, away in TODAYS_GAMES:
    home_week = teams[home].index.max()
    away_week = teams[away].index.max()

    print(f"{home}: using week {home_week}, {away}: using week {away_week}")

    home_df = teams[home].loc[home_week]
    away_df = teams[away].loc[away_week]

    # Prefix the opponent columns
    away_prefixed = away_df.add_prefix("opp_")

    # Merge into one row
    matchup = pd.concat([home_df, away_prefixed])
    rows.append(matchup)

X_pred = pd.DataFrame(rows)[feature_cols]
pred_spreads = model.predict(X_pred)
pred_winners = []
for (home, away), spread in zip(TODAYS_GAMES, pred_spreads):
    if spread > 0:
        pred_winners.append(home)
    else:
        pred_winners.append(away)
        
for (home, away), spread, winner in zip(TODAYS_GAMES, pred_spreads, pred_winners):
    logger.info(f"{home} vs {away} → predicted spread: {spread:.2f} → winner: {winner}")
