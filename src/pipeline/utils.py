import pickle
import mlflow
import mlflow.data
import pandas as pd
import mlflow.sklearn
import mlflow.catboost
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import dagshub

def save_object(file_path, obj, step):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f"{step} object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        # Initialize MLflow
        dagshub.init(repo_owner='suhaibmukhtar', repo_name='StudentPerformancePrediction', mlflow=True)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("ExperimentTrackingHyperTune")
        
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Starting evaluation for {model_name}")
            
            # Get corresponding parameters
            parameters = param[model_name]
            gs = GridSearchCV(model, parameters, cv=5)
            
            with mlflow.start_run(run_name=model_name):
                try:
                    # Log model name and dataset
                    mlflow.log_param("Model", model_name)
                    
                    # Log input data
                    train = pd.DataFrame(x_train).copy()
                    test = pd.DataFrame(x_test).copy()
                    train['target'] = y_train
                    test['target'] = y_test
                    train_df = mlflow.data.from_pandas(train)
                    test_df = mlflow.data.from_pandas(test)
                    mlflow.log_input(train_df, context="training")
                    mlflow.log_input(test_df, context="testing")

                    # Perform GridSearch
                    logging.info(f"Starting GridSearchCV for {model_name}")
                    gs.fit(x_train, y_train)
                    
                    # Log best parameters
                    mlflow.log_params(gs.best_params_)
                    
                    # Get best model and fit
                    best_model = gs.best_estimator_
                    
                    # Make predictions
                    y_test_pred = best_model.predict(x_test)
                    
                    # Calculate metrics
                    test_model_score = r2_score(y_test, y_test_pred)
                    test_rmse = root_mean_squared_error(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "R2_score": test_model_score,
                        "RMSE": test_rmse,
                        "MAE": test_mae
                    })

                    # Log model
                    if "CatBoost" in model_name:
                        mlflow.catboost.log_model(best_model, artifact_path=model_name)
                    else:
                        mlflow.sklearn.log_model(best_model, artifact_path=model_name)

                    # Save score in report
                    report[model_name] = test_model_score
                    
                    # Log source file and tags
                    mlflow.log_artifact(__file__)
                    mlflow.set_tags({
                        'Author': 'Suhaib Mukhtar',
                        "Project": 'StudentPerformance'
                    })
                    
                    logging.info(f"{model_name} evaluation completed - R2 Score: {test_model_score}")
                
                except Exception as e:
                    logging.error(f"Error during {model_name} evaluation: {str(e)}")
                    continue

        return report

    except Exception as e:
        raise CustomException(e, sys)