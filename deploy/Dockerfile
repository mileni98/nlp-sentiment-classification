FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

# Run nltk.download('stopwords'), nltk.download('popular')
RUN python -m nltk.downloader stopwords popular

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]