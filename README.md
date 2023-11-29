MUST DO apply clearning functions to input of ml 
# Extract the predicted class probabilities
predicted_probabilities = torch.nn.functional.softmax(model_output.logits, dim=-1).squeeze().tolist()
add to see probabilities

# Run
- position in deploy folder
- docker build -t app .
- docker run -p 8000:8000 --env-file .env app


# To-Do List
- Review falsely classified samples.
- Enhance TfIdf values for optimal parameters.
- Increase n_iter for random forest and adjust verbosity settings.
- Implement an inference pipeline for text request-response.


# Access Server via SSH
- ssh beta@192.168.66.221
- cd projects/praksa/aleksa, ll - list files, pwd - print working directory

- Open new Powerschell window and copy: scp -r /path/to/folder/* beta@192.168.66.221:/path/to/remote/folder/
- scp -r D:\nlp-project-server/* beta@betaserver:~/projects/praksa/aleksa/

scp -r beta@betaserver:~/projects/praksa/aleksa/* D:/nlp-project-server/



# Setup virtual environment
- workon - list 
- mkvirtualenv praksa_env
- rmvirtualenv  
- Read documentation - https://virtualenvwrapper.readthedocs.io/en/latest/

- pip install -r requirements.txt

# Run python script
- python bert.py


# Monitor GPU and adjust parameters
- nvidia-smi
- watch -n5 nvidia-smi


# WSL 2 docker
- wsl --list --online
- wsl --install    (aleksa, 1234)
- wsl --status 

# Ubuntu
- previous will install ubuntu command line, install in visual studio WSL extension to acces wsl terminal
- in ubuntru type:
- sudo apt-get update
- sudo apt-get install docker-compose-plugin

# Fix 
- Move wsl do D drive - https://www.youtube.com/watch?v=13jo3ppi7a0&ab_channel=TroubleChute
- wsl --list
- wsl --shutwodn
- wsl

# Installation 
- pip install contractions, pyspellchecker, ipynb, mlflow, xgboost, scikit-optimize, numpy, pandas, scikit-learn, transformers


# Modification Required
- replaced all np.int with int in the file 'anaconda3\envs\myenv\Lib\site-packages\skopt\space\transformers.py'

# Setup NLTK to enable word tokenization
- nltk.download('popular') 

# Make mlflow server locally
- mlflow server --host 127.0.0.1 --port 8080

# Problems
- xgboost nije radio hyperparameter tunning , pokusano sa manjom dubinom stabal, ali nije radilo pa su postavljeni defaultni parametri
- loadovanje dl artifakta na mlflow - stavljena skripta na server i odatle pristupljeno mlflowu (install libraries)
- pracenje graficke kartice i opterecenosti i u skladu s itm povecavan batch size
- docker desktop dne sme da se koristi - wsl2 i skunit docker tamo, pa rad sa wsl ubuntu terminalom
- wsl puni memoriju - napravljen na du folder i fizicka lokacija kopirana da puni njega
