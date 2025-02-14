import pickle
from src.logger import logging
from src.exception import CustomException
import sys

def save_object(file_path, preprocessor_obj):
    try:
        pickle.dump(preprocessor_obj,open(file_path,'wb'))
        logging.info("Proprocessor-object-saved-successfully")
    except Exception as e:
        raise CustomException(e,sys)