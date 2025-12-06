import pandas as pd
from pathlib import Path

from nfl_prediction.config import INJURIES_PROC_DIR

def main():
    INJURIES_PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INJURIES_PROC_DIR / "cbs_injuries_cleaned_week14.csv")
    
    df["status"] = df["severity"].astype(str).str.lower().str.strip()
    
    df["is_starter"] = 1
    
    out = df[["season", "week", "team", "position_group", "status", "is_starter"]]
    
    out_path = Path("data/processed/injuries/injuries_week14_curated.csv")
    out.to_csv(out_path, index=False)
    
    print(f"Saved curated injury file to {out_path}")

if __name__ == "__main__":
    main()