# datasetgen container - Section 5.3
# Deterministic/semantic/hard question generation, groundedness and schema
# validation, and output file creation all happen inside this container.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY question_schema.json ./

# Default command: print a help message. Real production/pipeline commands
# are supplied via `command:` in docker-compose.yml or `docker compose exec`.
CMD ["python3", "-c", "print('datasetgen ready. Example: docker compose exec datasetgen python3 main.py check-ollama')"]
