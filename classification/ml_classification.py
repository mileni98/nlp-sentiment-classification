# ml_classification.py - Script for training and evaluating machine learning classifiers and logging them to MLFlow
import os
import mlflow
import mlflow
import pandas as pd

from config import *
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def set_mlflow_experiment(tracking_url, experiment_name):
    """Set up the mlflow experiment"""

    # Set the tracking uri to the mlflow server
    mlflow.set_tracking_uri(tracking_url)

    # Set the current active experiment and return the experiment metadata
    return mlflow.set_experiment(experiment_name)


def calculate_metrics(y_pred, y_test):
    """Calculate various classification metrics"""

    # Calculate accuracy and balanced accuracy used for inbalanced datasets
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)

    # Calculate metrics for negative class
    precision_neg = precision_score(y_test, y_pred, pos_label = 0)
    recall_neg = recall_score(y_test, y_pred, pos_label = 0)
    f1_neg = f1_score(y_test, y_pred, pos_label = 0)

    # Calculate metrics for positive class
    precision_pos = precision_score(y_test, y_pred, pos_label = 1)
    recall_pos = recall_score(y_test, y_pred, pos_label = 1)
    f1_pos = f1_score(y_test, y_pred, pos_label = 1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract TP, FP, TN, FN from the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    return accuracy, accuracy_balanced, precision_neg, recall_neg, f1_neg, precision_pos, recall_pos, f1_pos, tp, fp, tn, fn


def log_metrics_to_mlflow(metrics):
    """Log classification metrics to mlflow"""

    # Log accuracy
    mlflow.log_metric('accuracy', metrics[0])
    mlflow.log_metric('accuracy_balanced', metrics[1])

    # Log metrics for negative classes
    mlflow.log_metric('precision_neg', metrics[2])
    mlflow.log_metric('recall_neg', metrics[3])
    mlflow.log_metric('f1_neg', metrics[4])

    # Log metrics for positive classes
    mlflow.log_metric('precision_pos', metrics[5])
    mlflow.log_metric('recall_pos', metrics[6])
    mlflow.log_metric('f1_pos', metrics[7])

    # Log the confusion matrix elements
    mlflow.log_metric('tp', metrics[8])
    mlflow.log_metric('fp', metrics[9])
    mlflow.log_metric('tn', metrics[10])
    mlflow.log_metric('fn', metrics[11])


def log_model_to_mlflow(classifier, X_train, run_name):
    """Log the trained model to mlflow"""

    # Infer the model signature, show expected input and output for the model
    signature = infer_signature(X_train, classifier.predict(X_train))

    # Log the model to mlflow
    mlflow.sklearn.log_model(
        sk_model = classifier,
        artifact_path = ARTIFACT_PATH,
        signature = signature,
        input_example = X_train,
        registered_model_name = run_name,
    )


def vectorize_data(X_train, X_test, max_feature_no):
    """Vectorize data using Tf-Idf"""    

    # Vectorize train and test data using Tf-Idf
    vectorizer = TfidfVectorizer(max_features = max_feature_no)
    X_train_tf = vectorizer.fit_transform(X_train)

    # Test data doesn't need fitting
    X_test_tf = vectorizer.transform(X_test)
    
    return X_train_tf, X_test_tf


def prepare_dataset(csv_name, path, tfidf_max_feature_no):
    """Prepare and split the dataset into training and test sets"""

    # Read the excel file into the corresponding DataFrame
    dataset = pd.read_csv(path + csv_name)

    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(dataset['cleaned_review'], dataset['sentiment'], test_size = 0.2, random_state = 43)

    # Vectorize data using Tf-Idf
    X_train_tr, X_test_tr = vectorize_data(X_train, X_test, tfidf_max_feature_no)   

    return X_train_tr, y_train, X_test_tr, y_test


def hyperparameter_tune_cv(classifier_name, param_space, X_train, y_train, jobs):
    """Perform hyperparameter tunning using BayesSearchCV and return the best performing model"""

    # Define cross-validation split while remaining same class balance
    kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

    # Initialize BayesSearchCV for hyperparameter tuning, change verbose to 1 to see output info
    search = BayesSearchCV(estimator = classifier_name(), search_spaces = param_space, n_iter = 100, cv = kf, scoring = 'f1', verbose = 3, n_jobs = jobs, random_state = 42)

    # Perform hyperparameter tuning
    search.fit(X_train, y_train)

    # Retrieve the best F1 score obtained during hyperparameter tuning
    best_score = search.best_score_   

    # Create a classifier instance using the best parameters
    classifier = classifier_name(**search.best_params_)

    return classifier, best_score


def ml_test_and_log(classifier, data_name, tfidf_feature_no, X_train, y_train, X_test, y_test):
    """Train the model, evaluate on the test set, and log relevant information to mlflow"""
    
    # Make custom run names for each run to differentiate them on mlflow
    run_name = f"{classifier.__class__.__name__}_{os.path.splitext(data_name)[0]}_{tfidf_feature_no}"

   # Start an MLflow run using the previously defined name
    with mlflow.start_run(run_name = run_name):

        # Log the hyperparameters
        mlflow.log_params(classifier.get_params())
        
        # Log additional useful parameters
        mlflow.log_param('data_name', data_name)
        mlflow.log_param('tfidf_features', tfidf_feature_no)

        # Train the model
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_pred, y_test)
        
        # Log metrics to mlflow
        log_metrics_to_mlflow(metrics)

        # Log the model to mlflow
        log_model_to_mlflow(classifier, X_train, run_name)       

        # End the MLflow run
        mlflow.end_run()


# Main entry point of the script 
if __name__ == "__main__":

     # Set up the mlflow experiment
    set_mlflow_experiment(TRACKING_URL, EXPERIMENT_NAME_ML)

    # Define models and update name field in classifier list which containts their corresponding parameters
    CLASSIFIERS_LIST[0]['name'] = LogisticRegression
    CLASSIFIERS_LIST[1]['name'] = XGBClassifier
    CLASSIFIERS_LIST[2]['name'] = RandomForestClassifier

    # Iterate over each classifier in the list
    for classifier_element in CLASSIFIERS_LIST:

        # Extract classifier name and hyperparameter space
        classifier_name = classifier_element['name']
        jobs = classifier_element['jobs']
        param_space = classifier_element['param_space']

        # Iterate over each dataset in the list
        for csv_name in DATASETS_LIST:

            # Iterate over each value of max_feature tfidf
            for no_feature in TFIDF_FEATURES_LIST:

                # Prepare the dataset using current CSV file and TF-IDF vectorize it using number of features
                X_train_tr, y_train, X_test_tr, y_test = prepare_dataset(csv_name, DATA_PATH, no_feature)

                # Check if it's XGBoost, if it is don't do hyperparameter tunning as there is only one combination of parameters
                if classifier_name == XGBClassifier:

                    # Instantiate the classifier directly
                    classifier = classifier_name()
                else:
                    # Perform hyperparameter tuning using cross validation for the current classifier and get the best parameters
                    classifier, best_f1_score = hyperparameter_tune_cv(classifier_name, param_space, X_train_tr, y_train, jobs)

                # Train the classifier and test it on the test dataset while logging relevant information
                ml_test_and_log(classifier, csv_name, no_feature, X_train_tr, y_train, X_test_tr, y_test)