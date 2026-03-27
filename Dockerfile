FROM python:3.11-slim

# Install only the packages that actually exist in Debian
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY web ./web

# Create directory for runtime files
RUN mkdir -p /data

# Expose port
EXPOSE 8099

# Run the application
CMD ["python", "-u", "app.py"]