# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The shared analysis package is staged into this directory by
# scripts/deploy_job.sh (it lives at the repository root as aamos_concordance/).
COPY aamos_concordance/ ./aamos_concordance/
COPY main.py .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
