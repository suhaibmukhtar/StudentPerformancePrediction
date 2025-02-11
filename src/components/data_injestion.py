import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import sys
from src.logger import logging
from src.exception import CustomException

logging.info("Started Data-Injestion")
try:
    a=1/0
except Exception as e:
    raise CustomException(e,sys)