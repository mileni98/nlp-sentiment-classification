# To-Do List
- Review falsely classified samples.
- Enhance TfIdf values for optimal parameters.
- Increase n_iter for random forest and adjust verbosity settings.
- Implement an inference pipeline for text request-response.


# Access Server via SSH
- ssh beta@192.168.66.221
- cd projects/praksa/aleksa, ll - list files, pwd - print working directory

- Open new Powerschell window and copy: scp -r /path/to/folder/* beta@192.168.66.221:/path/to/remote/folder/


# Setup virtual environment
- workon - list 
- rmvirtualenv  
- Read documentation - https://virtualenvwrapper.readthedocs.io/en/latest/

- pip install -r requirements.txt


# Run python script
- python bert.py


# Monitor GPU and adjust parameters
- nvidia-smi
- watch -n5 nvidia-smi



# Installation 
- pip install contractions, pyspellchecker, ipynb, mlflow, xgboost, scikit-optimize, numpy, pandas, scikit-learn, transformers


# Modification Required
- replaced all np.int with int in the file 'anaconda3\envs\myenv\Lib\site-packages\skopt\space\transformers.py'

# Setup NLTK to enable word tokenization
- nltk.download('popular') 

# Make mlflow server locally
- mlflow server --host 127.0.0.1 --port 8080
