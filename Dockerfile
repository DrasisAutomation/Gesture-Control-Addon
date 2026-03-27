FROM python:3.11-slim

# Install system dependencies for OpenCV, MediaPipe, and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswscale-extra \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Optional: Install hardware acceleration drivers (for Intel/AMD GPUs)
# Uncomment if you have compatible hardware
# RUN apt-get update && apt-get install -y \
#     intel-media-va-driver \
#     vainfo \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY web ./web

# Create directory for runtime files
RUN mkdir -p /data

# Expose port for web UI and WebSocket
EXPOSE 8099

# Run the application
CMD ["python", "-u", "app.py"]