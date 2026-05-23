FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

RUN pip install --no-cache-dir -U pip

COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# include the artifacts + kb so the API works immediately
COPY artifacts /app/artifacts
COPY docs /app/docs
COPY data /app/data

RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["uvicorn", "energy_ai.api.main:app", "--host", "0.0.0.0", "--port", "8080"]