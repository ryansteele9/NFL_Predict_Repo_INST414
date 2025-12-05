import pandas as pd

STATUS_MAP = {
    "out": 1.0,
    "doubtful": 1.0,
    "questionable": 0.2,
    "ir": -0.35,
    "pup": -0.2,
}


POS_WEIGHT = {
    "QB": 5.0,
    "OL": 1.0,
    "WR_TE": 0.9,
    "RB": 0.4,
    "DB": 0.8,
    "FRONT7": 0.6,
}

def compute_team_injury_adjustments(inj_df: pd.DataFrame) -> pd.DataFrame:
    df = inj_df.copy()
    
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    
    df = df[df["is_starter"] == 1]
    
    df["pos_weight"] = df["position_group"].map(POS_WEIGHT).fillna(0.0)
    df["status_weight"] = df["status"].map(STATUS_MAP).fillna(0.0)
    
    df["inj_adjust"] = df["pos_weight"] * df["status_weight"]
    
    adj = (
        df.groupby(["season", "week", "team"], as_index=False)["inj_adjust"].sum()
    )
    
    adj["inj_adjust"] = -adj["inj_adjust"]
    return adj

def apply_injury_adjustments(pred_df: pd.DataFrame, team_adj: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    
    df = df.merge(
        team_adj.rename(columns={"team": "home_team", "inj_adjust": "inj_adj_team"}),
        on=["season", "week", "home_team"],
        how="left",
    )
    
    df = df.merge(
        team_adj.rename(columns={"team": "away_team", "inj_adjust": "inj_adj_opp"}),
        on=["season", "week", "away_team"],
        how="left",
    )
    
    df["inj_adj_team"] = df.get("inj_adj_team", 0.0).fillna(0.0)
    df["inj_adj_opp"]  = df.get("inj_adj_opp", 0.0).fillna(0.0)

    df["point_diff_adj"] = df["pred_point_diff"] + df["inj_adj_team"] - df["inj_adj_opp"]
    
    return df