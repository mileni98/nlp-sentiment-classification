# functions_dl.py - Contains functions needed for dl model prediction with fast api
import os
import torch
import mlflow
from transformers import BertTokenizer


def dl_tokenize_data(sentence, checkpoint):
    """Tokenize data using a specified tokenizer"""

    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    # Return tokenized data as a pytorch tensor, allowing the tokenized input to be directly used as an input to the PyTorch model
    return tokenizer(sentence, truncation=True, padding='max_length', return_tensors='pt')


def dl_load_model(model_name):
    """Load the model from specified folder name"""

    # Get the path to the model
    model_uri = f"{os.getcwd()}/{model_name}"

    # Load the model from the specified folder and map it to CPU, can be changed to gpu
    return mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu')) 


def dl_predict_sentiment(sentence, model_name, checkpoint):
    """Predict sentiment for the input sentence"""

    # Preprocess the sentence using the tokenizer
    tokenized_input = dl_tokenize_data(sentence, checkpoint)

    # Load the model
    loaded_model = dl_load_model(model_name)

    # Make prediction using the loaded model without gradient computation
    with torch.no_grad():
        model_output = loaded_model(**tokenized_input)

    # Get the predicted sentiment class and map predictions to 'positive' or 'negative'
    sentiment = 'positive' if torch.argmax(model_output.logits).item() == 1 else 'negative'

    return sentiment
