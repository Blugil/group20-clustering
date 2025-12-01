# Use a slim Python base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and use unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends g++

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

WORKDIR /app/server

# Start Gunicorn with your Flask app factory
# Assumes app/__init__.py has `def create_app(): ...`
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "server:create_app()"]
