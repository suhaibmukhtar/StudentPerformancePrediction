import pandas as pd
import sys
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#dataclass is a decorator with help of which we are able to directly define class-variables
#datainjestionconfig will contain all paths where we want to save our output
#Note: Only possible when we have to define variables only, nor the method, should contain variables only
@dataclass
class DataInjestionConfig:
    train_dataset_path=os.path.join('artifacts','train.csv')
    test_dataset_path=os.path.join('artifacts','test.csv')
    raw_dataset_path=os.path.join('artifacts','data.csv')
    
class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()
    
    def initiate_data_injestion(self):
        """
        This function will initiate data-injestion
        """
        logging.info("Started Data-Injestion-initiation")
        try:
            data = pd.read_csv(r"D:\ML\MLOPS\notebook\data\StudentPerformanceFactors.csv")
            logging.info("Dataset Loaded Successfully")
            #dirname is used to read the directory-name
            os.makedirs(os.path.dirname(self.injestion_config.train_dataset_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.injestion_config.test_dataset_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.injestion_config.raw_dataset_path),exist_ok=True)
            #after creation of directory saving the csv files
            data.to_csv(self.injestion_config.raw_dataset_path,index=False,header=True)
            logging.info("Train-test-split-initiated")
            train_dataset, test_dataset = train_test_split(data, test_size = 0.2, random_state = 42)
            train_dataset.to_csv(self.injestion_config.train_dataset_path,index=False,header=True)
            test_dataset.to_csv(self.injestion_config.test_dataset_path, index=False, header=True)
            logging.info("Train-test-split performed successfully")
            logging.info(f"Train-dataset shape:{train_dataset.shape}")
            logging.info(f"Test-dataset shape:{test_dataset.shape}")
            return(
                self.injestion_config.train_dataset_path,
                self.injestion_config.test_dataset_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj1 = DataInjestion()
    train_path, test_path= obj1.initiate_data_injestion()






