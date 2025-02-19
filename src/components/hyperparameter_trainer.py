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

@dataclass
class HyperTrainerConfig:
    Hypertuned_model_file_path = os.path.join("artifacts", 'hyper_tuned_model.pkl')
    
class ModelHyperTrainer:
    def __init__(self):
        self.hyper_trained_config = HyperTrainerConfig()
        
    def initiate_hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("---------------Started-Hyper-parameter-tuning-initiation--------------")
            models = {
                "Random-Forest": RandomForestRegressor(random_state=42),
                "GBoost": GradientBoostingRegressor(random_state=42),
                "AdBoost": AdaBoostRegressor(random_state=42),
                "CatBoost": CatBoostRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "KNN": KNeighborsRegressor()
            }
            params = {
                "Random-Forest": {
                    'n_estimators': [90, 110, 120, None],
                    'max_depth': [6, 8, 10, None],
                    'max_features': ['sqrt', 'log2', None],
                    'random_state': [42]
                },
                "GBoost": {
                    'n_estimators': [80, 110, 130, None],
                    'max_depth': [7, 9, 11, None],
                    'max_features': ['sqrt', 'log2', None],
                    'random_state': [42]
                },
                "AdBoost": {
                    'n_estimators': [60, 120, 130, None],
                    'random_state': [42]
                },
                'CatBoost': {  # Changed from CBoost to match models dict
                    'depth': [4, 7, 10, None],
                    'learning_rate': [0.001, 0.03, 0.1],
                    'iterations': [50, 100, 150],
                    'random_state': [42]
                },
                'XGBoost': {  # Changed from Xgboost to match models dict
                    'learning_rate': [0.01, 0.001, 0.1],
                    'n_estimators': [20, 50, 80, 120, None],
                    'random_state': [42]
                },
                "DecisionTree": {  # Changed from DTree to match models dict
                    'criterion': ["gini", "entropy", "log_loss"],
                    'max_depth': [4, 7, 12, None],
                    'random_state': [42]
                },
                "KNN": {  # Changed from KNNR to match models dict
                    'n_neighbors': [4, 8, 9, 12],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }
            
            logging.info("Started Model Training and evaluation")
            model_report: dict = evaluate_models(x_train=X_train, y_train=y_train, 
                                              x_test=X_test, y_test=y_test,
                                              models=models, param=params)
            logging.info("Report obtained Successfully")
            
            # Getting the best model from report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(f"Best Model Score: {best_model_score}")
            
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)
            else:
                logging.info("Best Model Obtained Successfully!")
                save_object(
                    file_path=self.hyper_trained_config.Hypertuned_model_file_path,  # Fixed attribute reference
                    obj=best_model,
                    step='Trained_Model'
                )
            
            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)

# #will contain all the training-code and all models
# from sklearn.ensemble import (
#     RandomForestRegressor,
#     GradientBoostingRegressor,
#     AdaBoostRegressor,
# )
# import os
# import sys
# from src.logger import logging
# from src.exception import CustomException
# from dataclasses import dataclass
# from catboost import CatBoostRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from src.pipeline.utils import save_object, evaluate_models

# @dataclass
# class HyperTrainerConfig:
#     Hypertuned_model_file_path = os.path.join("artifacts",'hyper_tuned_model.pkl')
    
# class ModelTrainer:
#     def __init__(self):
#         self.hyper_trained_config = HyperTrainerConfig()
        
#     def initiate_hyperparameter_tuning(self,X_train,y_train,X_test,y_test):
#         try:
#             logging.info("---------------Started-Hyper-parameter-tuning-initiation--------------")
#             models ={
#                 "Random-Forest":RandomForestRegressor(random_state=42),
#                 "GBoost":GradientBoostingRegressor(random_state=42),
#                 "AdBoost":AdaBoostRegressor(random_state=42),
#                 "CatBoost":CatBoostRegressor(random_state=42),
#                 'XgBoost':XGBRegressor(random_state=42),
#                 "DecisionTree":DecisionTreeRegressor(random_state=42),
#                 "KNN":KNeighborsRegressor()
#             }
#             params={
#                     "Random-Forest":{
#                         'n_estimators':[90,110,120,None],
#                         'max_depth':[6,8,10,None],
#                         'max_features':['sqrt','log2',None],
#                         'random_state':42
#                     },
#                     "GBoost":{
#                         'n_estimators':[80,110,130,None],
#                         'max_depth':[7, 9, 11,None],
#                         'max_features':['sqrt','log2',None],
#                         'random_state':42
#                     },
#                     "AdBoost":{
#                         'n_estimators':[60,120,130,None],
#                         'random_state':42
#                     },
#                     'CBoost':{
#                         'depth':[4,7,10,None],
#                         'learning_rate':[0.001,0.03,0.1],
#                         'iterations':[50,100,150],
#                         'random_state':42
#                     },
#                     'Xgboost':{
#                         'learning_rate':[0.01,0.001,0.1],
#                         'n_estimators':[20,50,80,120,None],
#                         'random_state':42
#                     },
#                     "DTree":{
#                         'criterion':["gini", "entropy", "log_loss"],
#                         'max_depth':[4,7,12,None],
#                         'random_state':42
#                     },
#                     "KNNR":{
#                         'n_neighbors':[4,8,9,12],
#                         'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
#                     }
#             }
#             logging.info("Started Model Training and evaluation")
#             model_report: dict = evaluate_models(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models, param=params)
#             logging.info("Report obtained Successfully")
#             #getting the best model from report
#             best_model_score = max(sorted(model_report.values()))
#             #to get the best-model-name
#             best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]
#             best_model = models[best_model_name]
#             logging.info(f"Best Model Name:{best_model}")
#             if best_model_score<0.6:
#                 raise CustomException("No Best Model Found",sys)
#             else:
#                 logging.info("Best Model Obtained Successfully!")
#                 save_object(file_path=self.trained_config.trained_model_file_path, obj=best_model, step='Trained_Model')
            
#             return best_model_score
#         except Exception as e:
#             raise CustomException(e,sys)
        
    