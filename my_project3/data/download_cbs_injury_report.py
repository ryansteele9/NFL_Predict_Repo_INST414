import pandas as pd
from my_project3.config import INJURIES_RAW_DIR

URL = "https://www.cbssports.com/nfl/injuries/"

tables = pd.read_html(URL)

injury_like = []
for t in tables:
    cols = set(t.columns.astype(str))
    if {"Player", "Position", "Injury Status"}.issubset(cols):
        injury_like.append(t)

if not injury_like:
    raise RuntimeError("Could not find any injury tables; CBS layout may have changed.")

inj = pd.concat(injury_like, ignore_index=True)

def extract_severity(status_text: str) -> str:
    """
    Take status descriptions and extract game status (e.g. Questionable, Out).
    """
    s = str(status_text).lower()
    
    if "questionable" in s:
        return "questionable"
    if "doubtful" in s:
        return "doubtful"
    if "out" in s and "for week" in s:
        return "out"
    if "injured reserve" in s:
        return "ir"
    if "physically unable to perform" in s:
        return "pup"
    
    return "other"

inj["severity"] = inj["Injury Status"].apply(extract_severity)
inj_filtered = inj[inj["severity"].isin(["out", "doubtful", "questionable", "ir", "pup"])]

out_path = INJURIES_RAW_DIR / "CBS_injuries.csv"
inj_filtered.to_csv(out_path, index=False)