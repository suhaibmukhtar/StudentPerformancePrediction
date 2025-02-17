#will contain all the training-code and all models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.pipeline.utils import save_object, evaluate_models
from sklearn.metrics import r2_score, root_mean_squared_error
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.trained_config = ModelTrainerConfig()
        
    def initiate_model_training(self,X_train,y_train,X_test,y_test):
        try:
            logging.info("---------------Started-Model-trainer-initiation--------------")
            models ={
                "Random-Forest":RandomForestRegressor(random_state=42),
                "GBoost":GradientBoostingRegressor(random_state=42),
                "AdBoost":AdaBoostRegressor(random_state=42),
                "CatBoost":CatBoostRegressor(random_state=42),
                'XgBoost':XGBRegressor(random_state=42),
                "DecisionTree":DecisionTreeRegressor(random_state=42),
                "KNN":KNeighborsRegressor()
            }
            logging.info("Started Model Training and evaluation")
            model_report: dict = evaluate_models(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)
            logging.info("Report obtained Successfully")
            #getting the best model from report
            best_model_score = max(sorted(model_report.values()))
            #to get the best-model-name
            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best Model Name:{best_model}")
            if best_model_score<0.6:
                raise CustomException("No Best Model Found",sys)
            else:
                logging.info("Best Model Obtained Successfully!")
                save_object(file_path=self.trained_config.trained_model_file_path, obj=best_model, step='Trained_Model')
            
            return best_model_score
        except Exception as e:
            raise CustomException(e,sys)
        
    