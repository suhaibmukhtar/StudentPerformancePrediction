## code responsible for encoding/transforming the data
import pandas as pd
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_injestion import DataInjestionConfig
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts",'preprocessed_pipeline.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_transformation_object(self, numerical_cols, categorical_cols):
        """
        This function will return preprocessore object: only code related to pipeline
        """
        logging.info("---------started getting data-transformation object--------")
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ('numeric_impute',SimpleImputer(strategy='median')),
                    ('Scaling_numeric',StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ('categoric_impute',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoding',OneHotEncoder(sparse_output=False,drop='first')),
                    ('Scaler',StandardScaler())
                ]
            )
            preprocessor_pipeline = ColumnTransformer(
                transformers=[
                 ('num_pipeline',numerical_pipeline, numerical_cols),
                 ('categoric_pipeline',categorical_pipeline, categorical_cols)   
                ]
            )
            return preprocessor_pipeline
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path,test_path):
        """This function will initiate the data-transformation-process

        Args:
            train_path (_type_): Path to the training-data train_path
            test_path (_type_): Path to the testing-data
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Successfully read train and test data from artifacts")
            numerical_cols = list(train_data.select_dtypes(exclude="O").columns)
            categorical_cols = list(train_data.select_dtypes(include="O").columns)
            target_column="Exam_Score"
            numerical_cols.remove(target_column)
            logging.info("Obtaining the preporcessor object")
            preprocessor_obj = self.get_transformation_object(numerical_cols, categorical_cols)
            logging.info("Obtained Successfully Pre-processor object")
            X_train = train_data.drop(columns=['Exam_Score'])
            X_test = test_data.drop(columns=['Exam_Score'])
            y_train = train_data['Exam_Score']
            y_test = test_data['Exam_Score']
            ## Applying transformation
            X_train_trans = preprocessor_obj.fit_transform(X_train)
            X_test_trans = preprocessor_obj.transform(X_test)
            save_object(self.transformation_config.preprocessor_path,preprocessor_obj,step='Pre-processor')
            return(
                X_train_trans,
                y_train,
                X_test_trans,
                y_test,
                self.transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__=="__main__":
    obj2 = DataTransformation()
    obj3 = DataInjestionConfig()
    X_train,y_train,X_test,y_test, preprocessor_path=obj2.initiate_data_transformation(obj3.train_dataset_path,obj3.test_dataset_path)