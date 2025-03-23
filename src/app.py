from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
import os
from src.config import PREPROCESSOR_PKL_PATH, BEST_MODEL_PATH

# Initialize FastAPI App
app = FastAPI()

if os.path.exists(BEST_MODEL_PATH) and os.path.exists(PREPROCESSOR_PKL_PATH):
    with open(BEST_MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(PREPROCESSOR_PKL_PATH, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    print("Model & Preprocessor Loaded Successfully!")
else:
    raise FileNotFoundError("Model or Preprocessor file not found. Train the model first.")

# Define Input Schema
class InputFeatures(BaseModel):
    Hours_Studied: int
    Attendance: int
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: int
    Previous_Scores: int
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: int
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str

@app.get("/")
def home():
    return {"message": "MLOps FastAPI Model Deployed Successfully!"}

@app.post("/predict")
def predict(data: InputFeatures):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Transform input using the preprocessor
        transformed_input = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_input)[0]

        return {"predicted_score": float(prediction)}

    except Exception as e:
        return {"error": str(e)}

