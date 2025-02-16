import pickle
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score
def save_object(file_path, obj,step):
    try:
        pickle.dump(obj,open(file_path,'wb'))
        logging.info("Object-saved-successfully")
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
        
            y_test_pred = model.predict(x_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
            
            