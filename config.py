import sys
sys.dont_write_bytecode = True   

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"
EVIDENCE_DIR = BASE_DIR / "evidence"
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"


HOLD_COST = 380                              
CAPACITY = 40                                
POLICY_CHANGE = pd.Timestamp("2026-05-01")   
SMALL_LIMIT = 2000                           
NEW_COHORT_START = pd.Timestamp("2025-10-01")  

LAG_DAYS = 45          
PRIOR_RATE = 0.013     
PRIOR_STRENGTH = 20    
POST_WEIGHT = 3.0      
SEED = 42              