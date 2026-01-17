# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for numerical computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Make run_workflow.sh executable
RUN chmod +x run_workflow.sh

# Default workflow type (can be overridden via environment variable)
ENV WORKFLOW_TYPE=kp14

# Run workflow based on WORKFLOW_TYPE environment variable
CMD ["./run_workflow.sh"]
