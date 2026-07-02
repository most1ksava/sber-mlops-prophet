FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)" && \
    python -c "import importlib_resources, shutil, os; p = str(importlib_resources.files('prophet') / 'stan_model' / 'cmdstan-2.33.1'); shutil.rmtree(p, ignore_errors=True)"

ENV CMDSTAN_PATH=/root/.cmdstan/cmdstan-2.39.0

COPY model.py .
COPY app/ app/
COPY retrainer.py .
COPY mlflow_artifacts/ mlflow_artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
