# Use a lightweight base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port
EXPOSE 8000

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
