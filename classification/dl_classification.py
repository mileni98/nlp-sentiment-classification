# dl_classification.py - Script for training a BERT-based sentiment analysis model using Hugging Face Transformers and logging it to MLFlow
import mlflow
import numpy as np
import pandas as pd

from config import *
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, TrainingArguments, Trainer, DataCollatorWithPadding


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


def log_model_to_mlflow(model, run_name):
    """Log the trained model to mlflow"""

    # Log the model to mlflow using pytorch
    model_info = mlflow.pytorch.log_model(
        pytorch_model = model,
        artifact_path = ARTIFACT_PATH,
        registered_model_name = run_name,
    )

    return model_info


def prepare_and_split_dataset(csv_name, path):
    """Prepare and split the dataset into training, validation, and test sets"""

    # Read the excel file into a corresponding DataFrame
    dataset = pd.read_csv(path + csv_name)

    # Replace sentiment values with binary values
    dataset['sentiment'] = dataset['sentiment'].replace({'negative':0, 'positive':1})

    # Rename columns in the DataFrame
    dataset = dataset.rename(columns={'review' : 'text', 'sentiment' : 'labels'})

    # Split data into training and testing datasets
    train_data, test_data = train_test_split(dataset, test_size = 0.2, random_state = 43)

    # Further split training data into train and validation sets, 0.25 because first split is 37200 - 9300, and then 0.25 of 37200 is 9300
    train_data, val_data = train_test_split(train_data, test_size = 0.25, random_state = 43)

    # Create a Dataset dictionary and convert dataframes to datasets
    combined_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'val': Dataset.from_pandas(val_data),
        'test': Dataset.from_pandas(test_data)
    })

    return combined_dataset


def tokenize_data(combined_dataset, tokenizer):
    """Tokenize data using a specified tokenizer"""

    # Tokenize function
    def tokenize_function(examples):
        # max_length padding ensures same length of each sequence within a batch, truncation parameter doesn't play a role
        return tokenizer(examples['text'], truncation = True, padding = 'max_length')

    # Tokenize the combined dataset, batched parameter tokenizes multiple samples simultaneously
    tokenized_dataset = combined_dataset.map(tokenize_function, batched = True)

    # Uncomment to return split datasets
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['val']
    test_dataset = tokenized_dataset['test']

    return train_dataset, eval_dataset, test_dataset


def initialize_trainer_and_model(train_dataset, eval_dataset, tokenizer, training_args):
    """Initialize the Trainer and the model"""

    # Create a data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Load the pretrained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels = 2)
    
    # Initialize the Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )
    
    return trainer, model


def ml_test_and_log(trainer, model, data_name, test_dataset):
    """Train the model, evaluate on the test set, and log relevant information to mlflow"""
        
    # Start an MLflow run using the previously defined name
    with mlflow.start_run(run_name = 'bert_model_v1'):
      
        # Log additional useful parameters
        mlflow.log_param('data_name', data_name)

        # Train the model using the pretrained trainer
        trainer.train()

        # Make predictions on the test dataset, y_pred contains list of probabilities for each sentiment class 
        # For example - [0.85, 0.15] Probability for positive sentiment is 85% and for negative is 15%
        y_pred = trainer.predict(test_dataset)
        
        # Extract predicted lables, get the class with highest probability
        preds = np.argmax(y_pred.predictions, axis=-1)
        
        # Calculate metrics by comparing real label and predicted label
        metrics = calculate_metrics(y_pred.label_ids, preds)
        
        # Log metrics to mlflow
        log_metrics_to_mlflow(metrics)
        
        # Log the model to mlflow
        log_model_to_mlflow(model, 'bert')

        # End the MLflow run
        mlflow.end_run()


# Main entry point of the script 
if __name__ == "__main__":

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        logging_dir = LOGGING_DIR,
        logging_steps = LOGGING_STEPS,
        save_steps = SAVE_STEPS,
        save_total_limit = SAVE_TOTAL_LIMIT,
        load_best_model_at_end = LOAD_BEST_MODEL_AT_END,
        metric_for_best_model = METRIC_FOR_BEST_MODEL,
        evaluation_strategy = EVALUATION_STRATEGY,
        save_strategy = SAVE_STRATEGY,
    )

    # Set up the mlflow experiment
    set_mlflow_experiment(TRACKING_URL, EXPERIMENT_NAME_DL)

    # Preprocess dataset and split into training, evaluation and test
    combined_dataset = prepare_and_split_dataset(FILE_NAME_DL, DATA_PATH)

    # Tokenize data from the dataset
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)
    train_dataset, eval_dataset, test_dataset = tokenize_data(combined_dataset, tokenizer)

    # Initialize the trainer
    trainer, model = initialize_trainer_and_model(train_dataset, eval_dataset, tokenizer, training_args)

    # Train the classifier and test it on the test dataset while logging relevant information
    ml_test_and_log(trainer, model, FILE_NAME_DL, test_dataset)

  