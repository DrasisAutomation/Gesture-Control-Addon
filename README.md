# Gesture Control Add-on for Home Assistant v2.0

Control up to 5 smart devices using hand gestures via RTSP camera. Perfect for controlling lights, switches, fans, and other smart devices with simple hand gestures.

## Features

- 🖐️ **5-Device Control** - Control up to 5 different entities with individual finger gestures
- 🎯 **Accurate Gesture Detection** - Improved finger counting with confidence scoring
- 📹 **Low Latency Streaming** - Direct RTSP stream passthrough with no re-encoding
- 🔄 **Auto-Reconnection** - Automatically reconnects to both camera and Home Assistant
- 🎮 **Reset Gesture** - Configurable gesture to turn off all devices at once
- 📊 **Real-time Feedback** - Web UI with confidence meter and device status
- ⚡ **High Performance** - Optimized processing with configurable FPS and stability frames

## Gesture Mapping

| Gesture | Fingers | Action |
|---------|---------|--------|
| Fist | 0 | RESET: Turn OFF all configured devices |
| 1 Finger | 1 | Toggle Device 1 (on/off) |
| Victory | 2 | Toggle Device 2 (on/off) |
| OK | 3 | Toggle Device 3 (on/off) |
| 4 Fingers | 4 | Toggle Device 4 (on/off) |
| Palm | 5 | Toggle Device 5 (on/off) |

**Note:** The reset gesture (fist) can be configured to any finger count (0-5) in settings.

## Configuration

### Basic Settings

| Option | Description | Example |
|--------|-------------|---------|
| `rtsp_url` | RTSP camera URL | `rtsp://admin:password@192.168.1.100:554/stream` |
| `ha_url` | Home Assistant WebSocket URL | `ws://homeassistant.local:8123/api/websocket` |
| `ha_token` | Long-lived access token from Home Assistant | `eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...` |

### Device Entities (5 Configurable)

| Option | Description | Example |
|--------|-------------|---------|
| `entity_1` | Entity controlled by 1 finger | `light.bedroom_light` |
| `entity_2` | Entity controlled by 2 fingers | `light.living_room` |
| `entity_3` | Entity controlled by 3 fingers | `switch.fan` |
| `entity_4` | Entity controlled by 4 fingers | `light.kitchen` |
| `entity_5` | Entity controlled by 5 fingers | `cover.garage_door` |

### Performance Settings

| Option | Description | Range | Default |
|--------|-------------|-------|---------|
| `fps` | Target processing FPS | 5-30 | 15 |
| `frame_width` | Camera frame width | 320-1920 | 640 |
| `frame_height` | Camera frame height | 240-1080 | 480 |
| `bitrate` | Stream bitrate | any | 500k |

### Gesture Detection Settings

| Option | Description | Range | Default |
|--------|-------------|-------|---------|
| `detection_confidence` | MediaPipe detection confidence | 0.3-0.9 | 0.5 |
| `tracking_confidence` | MediaPipe tracking confidence | 0.3-0.9 | 0.5 |
| `cooldown_seconds` | Cooldown between same gestures | 0-5 | 1 |
| `reset_gesture` | Finger count that resets all devices | 0-5 | 0 |
| `stability_frames` | Frames needed for gesture stabilization | 3-10 | 5 |

### Example Configuration

```json
{
  "rtsp_url": "rtsp://admin:MyPassword@192.168.1.50:554/stream",
  "ha_url": "ws://homeassistant.local:8123/api/websocket",
  "ha_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "entity_1": "light.bedroom_lamp",
  "entity_2": "light.living_room_main",
  "entity_3": "light.kitchen_strip",
  "entity_4": "switch.garden_lights",
  "entity_5": "fan.ceiling_fan",
  "fps": 15,
  "frame_width": 640,
  "frame_height": 480,
  "cooldown_seconds": 1,
  "detection_confidence": 0.5,
  "tracking_confidence": 0.5,
  "reset_gesture": 0,
  "stability_frames": 5
}