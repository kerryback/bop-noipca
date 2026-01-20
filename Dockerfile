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

# Default entrypoint - can be overridden by --git-run-command in Koyeb
# Usage: Set MODEL, START, END env vars, or use --git-run-command
CMD ["python", "main.py", "kp14", "0", "1", "--koyeb"]
