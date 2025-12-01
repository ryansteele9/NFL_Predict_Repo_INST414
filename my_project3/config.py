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
RAW_DATA_DIR       = DATA_DIR / "raw"
INTERIM_DATA_DIR   = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR  = DATA_DIR / "external"
FEATURES_DATA_DIR  = DATA_DIR / "features"
MATCHUPS_DATA_DIR  = DATA_DIR / "matchups"
TEAMS_DATA_DIR     = DATA_DIR / "teams"
ODDS_RAW_DIR       = RAW_DATA_DIR / "odds"
ODDS_PROC_DIR      = PROCESSED_DATA_DIR / "odds"

MODELS_DIR         = PROJ_ROOT / "my_project3" / "modeling"

SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
