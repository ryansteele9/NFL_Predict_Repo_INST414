from pathlib import Path
import os

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR           = PROJ_ROOT / "data"

RAW_DIR            = DATA_DIR / "raw"
RAW_STATS_JSON     = RAW_DIR / "sdio_json"
RAW_STATS_CSV      = RAW_DIR / "team_stats_csv"
ODDS_RAW_DIR       = RAW_DIR / "odds"

EXTERNAL_DIR       = DATA_DIR/ "external"
INJURIES_RAW_DIR   = EXTERNAL_DIR / "injuries"

INTERIM_DATA_DIR   = DATA_DIR / "interim"

PROC_DIR           = DATA_DIR / "processed"
CLEAN_STATS_DIR    = PROC_DIR / "clean_team_stats_season"
TEAMS_DIR          = PROC_DIR / "teams"
FEATURES_DIR       = PROC_DIR / "features"
MATCHUPS_DIR       = PROC_DIR / "matchups"
ODDS_PROC_DIR      = PROC_DIR / "odds"
INJURIES_PROC_DIR  = PROC_DIR / "injuries"

MODELS_DIR         = PROJ_ROOT / "models"

SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
