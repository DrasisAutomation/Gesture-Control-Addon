ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11
FROM ${BUILD_FROM}

ENV LANG C.UTF-8

# Install system dependencies
RUN apk add --no-cache \
    ffmpeg \
    libstdc++ \
    mesa-gl \
    linux-headers

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY web/ ./web/
COPY run.sh .

RUN chmod a+x run.sh

CMD [ "/app/run.sh" ]
