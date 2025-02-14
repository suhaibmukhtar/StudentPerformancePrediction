import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(),"Logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
log_file_name=f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(filename)s-%(funcName)s-%(lineno)d-%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_file_name)),
        logging.StreamHandler()
    ]

)

if __name__=="__main__":
    logging.info("Logging Has Been Started")