FROM python:3.8-slim

WORKDIR /ml_training

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/pipeline/train_pipeline.py"]
