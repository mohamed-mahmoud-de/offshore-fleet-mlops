# Packaged app image: runs the pipeline modules and serves the Streamlit dashboard.
#   docker build -t offshore-fleet-app .
#   docker run --rm -p 8501:8501 --env-file .env \
#       -e POSTGRES_HOST=host.docker.internal offshore-fleet-app
# (the dashboard's predictor works standalone; the scorecard view needs the DB)
FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by LightGBM at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
COPY config ./config
COPY dags ./dags
COPY models ./models
COPY data ./data

ENV PYTHONPATH=/app \
    MLFLOW_ENABLED=0

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/serving/dashboard.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]
