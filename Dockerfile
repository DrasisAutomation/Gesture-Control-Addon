ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm
FROM ${BUILD_FROM}

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for Python, MediaPipe, OpenCV, and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (using pre-compiled binary wheels for speed)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary --break-system-packages -r requirements.txt

# Copy application files
COPY app.py .
COPY web/ ./web/
COPY run.sh /app/run.sh

RUN chmod a+x /app/run.sh

CMD [ "/app/run.sh" ]
