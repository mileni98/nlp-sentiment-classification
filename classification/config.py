# Set the name of the pretrained bert model that will be used
CHECKPOINT = 'bert-base-cased'

# Set parameters needed to connect to mlflow
TRACKING_URL = 'http://192.168.66.221:20002'
EXPERIMENT_NAME_DL = 'aleksa_praksa_dl'
EXPERIMENT_NAME_ML = 'aleksa_praksa'
ARTIFACT_PATH = 'artifact'

# Set folder path to the original dataset
DATA_PATH = './data/'
FILE_NAME_DL = 'imdb_dataset.csv'

# TrainingArguments configuration for bert
OUTPUT_DIR = 'output'
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 10
LOGGING_DIR = './logs'
LOGGING_STEPS = 1000
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 5
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = 'eval_loss'
EVALUATION_STRATEGY = 'steps'
SAVE_STRATEGY = 'steps'


# Dataset configuration
# datasets_list = [file for file in os.listdir('../data/') if file.startswith('split') and file.endswith('.csv')]
DATASETS_LIST = ['split_F_spel_F_lem.csv', 'split_F_spel_F_stem.csv', 'split_F_spel_T_lem.csv', 'split_F_spel_T_stem.csv', 'split_T_spel_F_lem.csv', 'split_T_spel_F_stem.csv', 'split_T_spel_T_lem.csv', 'split_T_spel_T_stem.csv']
TFIDF_FEATURES_LIST = [10000, 15000, 20000, 25000]

# Classifier configurations, name parameters will be set in main script
CLASSIFIERS_LIST = [
    {
        # LogisticRegression parameters
        'name': None,
        'jobs': -1,
        'param_space': {
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'C': (1e-4, 1e+4, 'log-uniform'), 
            'penalty': ['l2'],
            'max_iter': (100, 1000),
        }
    },
    {
        # XGBClassifier parameters
        'name': None,
        'jobs': 1,
        'param_space': {
            'n_estimators': [200],
            'learning_rate': [0.01],
            'max_depth': [15],
        }
    }, 
    {    
        # RandomForestClassifier parameters
        'name': None,
        'jobs': -1,
        'param_space': {
            'n_estimators': (10, 200),
            'max_features': ['sqrt', 'log2'], 
            'max_depth': (5, 15),
            'min_samples_split': (2, 5),
            'min_samples_leaf': (1, 3),
            'bootstrap': [True, False],
        }
    },  
]