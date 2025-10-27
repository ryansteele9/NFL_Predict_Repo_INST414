import pandas as pd

# variable table for example 2022 week 1 raw file
df = pd.read_csv("/Users/ryansteele/my_repo2/my_project3/data/raw/team_stats_2022REG_week1.csv")

variable_inventory = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing %": df.isna().mean().round(3) * 100
})

variable_inventory["Description"] = ""

variable_inventory.to_csv("reports/variable_inventory.csv", index=False)
print("Saved to reports/variable_inventory.csv")