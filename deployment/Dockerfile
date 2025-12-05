# Dockerfile for Background Replacement System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Set default environment variables
ENV LOG_LEVEL=INFO
ENV USE_CLOUD_LOGGING=true

# Expose port (for Cloud Run)
EXPOSE 8080

# Run the application
# For Cloud Run, use gunicorn or the function framework
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 540 deployment.main:background_replacement_handler



