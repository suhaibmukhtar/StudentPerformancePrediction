# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .
COPY src /app/src
COPY artifacts /app/artifacts

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "127.0.0.1", "--port", "8000"]
