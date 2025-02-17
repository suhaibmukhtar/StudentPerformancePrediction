import pickle
import mlflow
import mlflow.sklearn
import mlflow.catboost
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import dagshub
dagshub.init(repo_owner='suhaibmukhtar', repo_name='StudentPerformancePrediction', mlflow=True)

# Set MLflow tracking server URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("https://dagshub.com/suhaibmukhtar/StudentPerformancePrediction.mlflow")
#Name of experiment
mlflow.set_experiment("ExperimentTrackingStudentPerformance")

# Authenticate directly if needed
mlflow.tracking.MlflowClient().get_experiment_by_name("ExperimentTrackingStudentPerformance")

def save_object(file_path, obj, step):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate multiple models and log results in MLflow
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        for model_name, model in models.items():
            with mlflow.start_run():
                # Log model name as a parameter
                mlflow.log_param("Model", model_name)

                # Log model-specific parameters 
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())

                # Train the model
                model.fit(x_train, y_train)
                
                # Make predictions
                y_test_pred = model.predict(x_test)
                
                # Calculate R2 score
                test_model_score = r2_score(y_test, y_test_pred)
                test_rmse = root_mean_squared_error(y_test,y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                # Log the R2 score
                mlflow.log_metric("R2_score", test_model_score)
                mlflow.log_metric("RMSE", test_rmse)
                mlflow.log_metric("MAE",test_mae)

                # Log the model depending on its type
                if "CatBoost" in model_name:
                    mlflow.catboost.log_model(model, artifact_path=model_name)
                else:
                    mlflow.sklearn.log_model(model, artifact_path=model_name)

                # Save the model score in the report
                report[model_name] = test_model_score
                
                logging.info(f"{model_name} - R2 Score: {test_model_score}")
                mlflow.set_tags({'Author':'Suhaib Mukhtar',"Project":'SudentPerformance'})

        return report

    except Exception as e:
        raise CustomException(e, sys)
