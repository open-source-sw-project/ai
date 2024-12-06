# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 모델 파일들을 먼저 복사
COPY model/*.pth /app/model/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/4c_b5ns_1.5e_640_ext_15ep_best_fold0.pth

# Expose the application port
EXPOSE 5000

# Healthcheck for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Command to run the application
CMD ["python", "app.py"]