import os

from functions_ml import *
from functions_dl import *
from fastapi import FastAPI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Retrieving environment variables
CHECKPOINT = os.getenv('CHECKPOINT')
MODEL_NAME_DL = os.getenv('MODEL_NAME_DL')
TRACKING_URL = os.getenv('TRACKING_URL')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
MODEL_NAME_ML = os.getenv('MODEL_NAME_ML')
MODEL_VERSION_ML = os.getenv('MODEL_VERSION_ML')
CSV_PATH = os.getenv('CSV_PATH')
CSV_NAME = os.getenv('CSV_NAME')
TFIDF_MAX_FEATURES = os.getenv('TFIDF_MAX_FEATURES')

# FastAPI application, to start the app type 'uvicorn main:app --reload' in anaconda prompt
app = FastAPI()

# Setting up the mlflow experiment
set_mlflow_experiment(TRACKING_URL, EXPERIMENT_NAME)


# Make predictions using the best performing ml model - LogisticRegression
@app.post('/predict-ml')
def predict(text_input_ml: str):

    # Predict sentiment using ML model
    predicted_class_ml = ml_predict_sentiment(text_input_ml, MODEL_NAME_ML, MODEL_VERSION_ML, CSV_NAME, CSV_PATH, TFIDF_MAX_FEATURES)

    return {'sentiment': predicted_class_ml}


# Make predictions using the best performing dl model - bert
@app.post('/predict-dl')
def predict(text_input_dl: str):

    # Predict sentiment using DL model
    predicted_class_dl = dl_predict_sentiment(text_input_dl, MODEL_NAME_DL, CHECKPOINT)
        
    return {'sentiment': predicted_class_dl}

