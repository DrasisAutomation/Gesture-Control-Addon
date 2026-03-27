FROM python:3.11-slim

# Install minimal system dependencies for OpenCV headless and MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY run.sh .
COPY web ./web

# Make run script executable
RUN chmod +x run.sh

# Expose port
EXPOSE 8099

# Run the application
CMD ["./run.sh"]