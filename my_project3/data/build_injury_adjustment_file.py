import pandas as pd
from pathlib import Path

def main():
    input_path = Path("data/processed/injuries/cbs_injuries_cleaned_week14.csv")
    df = pd.read_csv(input_path)
    
    df["status"] = df["severity"].astype(str).str.lower().str.strip()
    
    df["is_starter"] = 1
    
    out = df[["season", "week", "team", "position_group", "status", "is_starter"]]
    
    out_path = Path("data/processed/injuries/injuries_week14_curated.csv")
    out.to_csv(out_path, index=False)
    
    print(f"Saved curated injury file to {out_path}")

if __name__ == "__main__":
    main()