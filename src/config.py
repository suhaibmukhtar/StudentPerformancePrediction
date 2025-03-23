import os
from pathlib import Path
import sys
ROOTPACKAGE = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(ROOTPACKAGE))

ARTIFACTS_PATH = os.path.join(ROOTPACKAGE,"artifacts")
PREPROCESSOR_PKL_PATH = os.path.join(ARTIFACTS_PATH,"preprocessed_pipeline.pkl")
BEST_MODEL_PATH = os.path.join(ARTIFACTS_PATH,'model.pkl')
