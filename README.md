# Gesture Control Add-on for Home Assistant

Control your lights using hand gestures via RTSP camera.

## Features

- Real-time gesture detection using MediaPipe
- H.264/H.265 RTSP streaming with no re-encoding (low latency)
- Home Assistant WebSocket integration
- Configurable entity IDs and cooldown
- Web interface with live preview
- Auto-reconnection for both camera and HA

## Gesture Mapping

| Gesture | Fingers | Action |
|---------|---------|--------|
| Fist | 0 | Turn ON strip light |
| Victory | 2 | Turn ON row light |
| OK | 3 | Turn OFF row light |
| Palm | 5 | Turn OFF strip light |

## Configuration

| Option | Description |
|--------|-------------|
| rtsp_url | RTSP camera URL (e.g., rtsp://admin:password@192.168.1.100:554/stream) |
| ha_url | Home Assistant WebSocket URL (default: ws://homeassistant.local:8123/api/websocket) |
| ha_token | Long-lived access token from Home Assistant |
| strip_light_entity | Entity ID for strip light (e.g., light.bed_light) |
| row_light_entity | Entity ID for row light (e.g., light.desk_light) |
| fps | Target processing FPS (5-30, lower = less CPU usage) |
| cooldown_seconds | Cooldown between same gestures (prevents spam) |
| detection_confidence | MediaPipe detection confidence (0.3-0.9) |
| tracking_confidence | MediaPipe tracking confidence (0.3-0.9) |

## Installation

1. Copy this folder to `/addons/gesture-control/` on your Home Assistant instance
2. Go to Settings → Add-ons → Local Add-ons
3. Click on "Gesture Control" and install
4. Configure the add-on with your RTSP URL and HA token
5. Start the add-on

## Access

Open the web UI at: `http://[home-assistant-ip]:8099`

## Troubleshooting

### HEVC errors in logs
- The add-on handles HEVC streams, but errors may appear if the stream is corrupted
- Try reducing camera resolution or switching to H.264 in camera settings

### No video in browser
- Check that RTSP URL is correct
- Ensure camera is accessible from Home Assistant
- Try accessing the stream directly with VLC to verify

### Gestures not detected
- Ensure good lighting conditions
- Hold hand in front of camera at a reasonable distance
- Adjust detection confidence settings in configuration

## License

MIT