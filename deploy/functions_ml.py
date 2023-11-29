# functions_ml.py - Contains functions needed for ml model prediction with fast api
import re
import mlflow
import contractions
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet


def set_mlflow_experiment(tracking_url, experiment_name):
    """Set up the mlflow experiment"""

    # Set the tracking uri to the mlflow server
    mlflow.set_tracking_uri(tracking_url)

    # Set the current active experiment and return the experiment metadata
    return mlflow.set_experiment(experiment_name)


def ml_clean_text(sentence):  
    """Clean a input by removing URLs, tags, non-word, non-whitespace characters, apostrophes, and underscores."""

    # Define regular expressions for cleaning
    url_expression, tags_expression, others_expression, apostrophe_expression, underscore_expression = r'https?://\S+', r'<.*?>', r'[^\w\s\d]', r'\'', r'_+'
        
    # Apply cleaning for URLs and tags
    sentence = re.sub(url_expression, ' ', sentence)
    sentence = re.sub(tags_expression, ' ', sentence)

    # Apply cleaning for non-word and non-whitespace characters
    sentence = re.sub(apostrophe_expression, '', sentence)  # e.g., "don't" -> "dont"
    sentence = re.sub(underscore_expression, ' ', sentence)
    
    return re.sub(others_expression, ' ', sentence)


def ml_remove_digits(sentence):
    """Remove digits from the input sentence."""

    # Define a regular expression to identify digits for cleaning
    digits_expression = r'[\d]'

    # Remove digits
    return re.sub(digits_expression, ' ', sentence)


def ml_tokenize_and_remove_stop(sentence, remove_stop, language = 'english'):
    """Tokenize and optionally remove stop words from the input string."""

    # Define language for a dictionary to be used for removal of stop words
    stop = stopwords.words(language)

    # Tokenize the input text
    tokens = word_tokenize(sentence)

    # Remove stop words if specified
    if remove_stop:
        tokens = [word for word in tokens if word.lower() not in stop]

    return tokens


def ml_get_wordnet_pos(treebank_tag):
    """Maps corresponding treebank tags to wordnet speech names."""

    # Map corresponding treebank tags so that it can be read by the Lemmatizer 
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Return noun as a default pos for lemmatization


def ml_lemmatization(tokens):
    """Lemmatize tokens with POS tagging."""

    # Initialize the WordNet lematizer
    lemmatizer = WordNetLemmatizer()

     # Applying lemmatization with POS tagging
    return [lemmatizer.lemmatize(word, pos = ml_get_wordnet_pos(pos_tag)) for word, pos_tag in pos_tag(tokens)]
    

def ml_preprocess_input(sentence):
    """Preprocess the input sentence for machine learning."""

    # Function that converts letters to lovercases
    sentence = sentence.lower()

    # Apply fix function to expand contractions
    sentence = contractions.fix(sentence)

    # Function that removes non-word and non-whitespace characters
    sentence = ml_clean_text(sentence)

    # Function that removes digits
    sentence = ml_remove_digits(sentence)

    # Tokenize and remove stop words
    tokens = ml_tokenize_and_remove_stop(sentence, True)

    # Lematize tokens with POS
    return ml_lemmatization(tokens)
    

def ml_load_model(model_name, model_version):
    """Load the model from mlflow using its name and version"""

    # Construct URI for logged mlflow model using the provided name and version
    model_uri = f"models:/{model_name}/{model_version}"

    # Load the mlflow model using the constructed URI
    loaded_model = mlflow.sklearn.load_model(model_uri)

    return loaded_model


def ml_vectorize_data(X_train, max_feature_no):
    """Fit the vectorizer using the training data"""
    
    # Initialize the vectorizer object
    vectorizer = TfidfVectorizer(max_features = max_feature_no)

    # Fit the vectorizer using training data
    _ = vectorizer.fit_transform(X_train)
    
    return vectorizer


def ml_prepare_vectorizer(csv_name, path, tfidf_max_feature_no):
    """Prepare vectorizer on training data on which models were trained"""

    # Read the CSV file into the corresponding DataFrame
    dataset = pd.read_csv(path + csv_name)

    # Split data into training and testing datasets
    X_train, _, _, _ = train_test_split(dataset['cleaned_review'], dataset['sentiment'], test_size = 0.2, random_state = 43)

    # Vectorize data using Tf-Idf
    vectorizer = ml_vectorize_data(X_train, tfidf_max_feature_no)   

    return vectorizer


def ml_predict_sentiment(sentence, model_name, model_version, csv_name, path, tfidf_max_feature_no):
    """Predict the sentiment of a sentence using a trained model."""

    # Preprocess the input sentence
    sentence = ml_preprocess_input(sentence)

    # Load the trained mlflow model
    loaded_model_ml = ml_load_model(model_name, model_version)

    # Fit the vectorizer on training data
    vectorizer = ml_prepare_vectorizer(csv_name, path, tfidf_max_feature_no)

    # Transform the new input using the vectorizer
    sentence_vect = vectorizer.transform([sentence])

    # Make predictions using the loaded model
    prediction = int(loaded_model_ml.predict(sentence_vect)[0])

    # Map predictions to 'positive' or 'negative'
    sentiment = 'positive' if prediction == 1 else 'negative'

    return sentiment


