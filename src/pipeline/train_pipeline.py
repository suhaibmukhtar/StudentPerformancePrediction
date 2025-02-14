#will contain code related to training the models
#Here we will import all the components need for training a model
from src.components.data_injestion import DataInjestion
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException

#pipeline
if __name__=="__main__":
    data_injestion = DataInjestion()
    train_path, test_path = data_injestion.initiate_data_injestion()
    logging.info("------------Data Injestion Successfully Executed---------------------------")
    data_transformation = DataTransformation()
    X_train,y_train,X_test,y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    logging.info("------------Date-Transformation Executed Successfully-----------------------")
    