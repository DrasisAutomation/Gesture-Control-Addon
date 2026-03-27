#!/usr/bin/env bashio

set -e

# Print startup message
echo "🎮 Gesture Control Add-on with H.264 Support"
echo "============================================"

# Check if FFmpeg is available
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg is installed: $(ffmpeg -version | head -n1)"
else
    echo "❌ FFmpeg is not installed! H.264 encoding will not work."
fi

# Check for hardware acceleration
if command -v vainfo &> /dev/null; then
    echo "✅ VAAPI drivers detected"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected"
fi

# Start the Python application
echo "🚀 Starting application..."
python /app/app.py