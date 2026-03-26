#!/usr/bin/env python3
"""
Gesture Control Add-on with RTSP Camera and Live Video Stream
- RTSP camera stream processing with MediaPipe Hands
- MJPEG video streaming to web browser
- WebSocket for real-time gesture updates
- Accurate finger counting
"""

import asyncio
import cv2
import numpy as np
import mediapipe as mp
import time
import json
import logging
import threading
import os
import sys
import base64
from aiohttp import web
import socketio
import httpx
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
RTSP_URL = ""
HA_URL = "ws://homeassistant.local:8123/api/websocket"
HA_TOKEN = ""
STRIP_LIGHT_ENTITY = "light.strip_light"
ROW_3_ENTITY = "light.row_3"
COOLDOWN_SECONDS = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

# Load configuration
try:
    config_path = "/data/options.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            RTSP_URL = config.get('rtsp_url', RTSP_URL)
            HA_URL = config.get('ha_url', HA_URL)
            HA_TOKEN = config.get('ha_token', HA_TOKEN)
            STRIP_LIGHT_ENTITY = config.get('strip_light_entity', STRIP_LIGHT_ENTITY)
            ROW_3_ENTITY = config.get('row_3_entity', ROW_3_ENTITY)
            COOLDOWN_SECONDS = config.get('cooldown_seconds', COOLDOWN_SECONDS)
            DETECTION_CONFIDENCE = config.get('detection_confidence', DETECTION_CONFIDENCE)
            TRACKING_CONFIDENCE = config.get('tracking_confidence', TRACKING_CONFIDENCE)
            FRAME_WIDTH = config.get('frame_width', FRAME_WIDTH)
            FRAME_HEIGHT = config.get('frame_height', FRAME_HEIGHT)
            FPS = config.get('fps', FPS)
            logger.info("✅ Loaded configuration")
            logger.info(f"   RTSP URL: {RTSP_URL[:50]}..." if RTSP_URL else "   RTSP URL: NOT CONFIGURED")
            logger.info(f"   HA URL: {HA_URL}")
            logger.info(f"   Strip Light: {STRIP_LIGHT_ENTITY}")
            logger.info(f"   Row 3: {ROW_3_ENTITY}")
except Exception as e:
    logger.error(f"Error loading config: {e}")

# Global state
current_frame = None
current_finger_count = 0
current_gesture_name = "No Hand"
current_gesture_icon = "🤚"
last_action = ""
detection_status = "no_hand"
camera_active = False
last_trigger_time = 0
last_gesture = None
frame_lock = threading.Lock()

# WebSocket server
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# HA WebSocket connection
ha_socket = None
ha_connected = False

# ============================================================================
# FINGER COUNTING
# ============================================================================

def count_extended_fingers(hand_landmarks):
    """Count extended fingers from MediaPipe landmarks"""
    if not hand_landmarks:
        return 0
    
    count = 0
    
    # Convert to list for easier access
    landmarks = []
    for i in range(21):
        landmarks.append(hand_landmarks.landmark[i])
    
    # Index finger
    if landmarks[8].y < landmarks[6].y - 0.03:
        count += 1
    
    # Middle finger
    if landmarks[12].y < landmarks[10].y - 0.03:
        count += 1
    
    # Ring finger
    if landmarks[16].y < landmarks[14].y - 0.02:
        count += 1
    
    # Pinky
    if landmarks[20].y < landmarks[18].y - 0.02:
        count += 1
    
    # Thumb
    is_right_hand = landmarks[5].x < landmarks[17].x
    if is_right_hand:
        if landmarks[4].x < landmarks[3].x - 0.02:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x + 0.02:
            count += 1
    
    return count

def get_gesture_info(finger_count):
    """Get gesture info based on finger count"""
    if finger_count == 0:
        return {'name': 'Fist', 'icon': '✊'}
    elif finger_count == 1:
        return {'name': '1 Finger', 'icon': '☝️'}
    elif finger_count == 2:
        return {'name': 'Peace', 'icon': '✌️'}
    elif finger_count == 3:
        return {'name': 'Three', 'icon': '👌'}
    elif finger_count == 4:
        return {'name': 'Four', 'icon': '🖖'}
    elif finger_count == 5:
        return {'name': 'Palm', 'icon': '✋'}
    else:
        return {'name': f'{finger_count} Fingers', 'icon': '🖐️'}

def draw_hand_landmarks(frame, landmarks, finger_count):
    """Draw hand landmarks and finger count on frame"""
    if not landmarks:
        return frame
    
    h, w = frame.shape[:2]
    
    # Draw connections
    connections = [
        [0,1], [1,2], [2,3], [3,4],  # Thumb
        [0,5], [5,6], [6,7], [7,8],  # Index
        [0,9], [9,10], [10,11], [11,12],  # Middle
        [0,13], [13,14], [14,15], [15,16],  # Ring
        [0,17], [17,18], [18,19], [19,20],  # Pinky
        [5,9], [9,13], [13,17]  # Palm
    ]
    
    for connection in connections:
        p1 = landmarks[connection[0]]
        p2 = landmarks[connection[1]]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw landmarks
    for i, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        is_tip = i in [4, 8, 12, 16, 20]
        color = (0, 255, 255) if is_tip else (0, 0, 255)
        radius = 6 if is_tip else 4
        cv2.circle(frame, (x, y), radius, color, -1)
    
    # Draw finger count
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw cooldown if active
    now = time.time()
    if last_trigger_time > 0 and now - last_trigger_time < COOLDOWN_SECONDS:
        remaining = COOLDOWN_SECONDS - (now - last_trigger_time)
        cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    return frame

# ============================================================================
# HOME ASSISTANT CONNECTION
# ============================================================================

async def connect_ha():
    """Connect to Home Assistant WebSocket"""
    global ha_socket, ha_connected
    
    while True:
        try:
            import websockets
            ha_socket = await websockets.connect(HA_URL)
            
            # Authenticate
            await ha_socket.send(json.dumps({
                "type": "auth",
                "access_token": HA_TOKEN
            }))
            
            response = await ha_socket.recv()
            data = json.loads(response)
            
            if data.get("type") == "auth_ok":
                ha_connected = True
                logger.info("✅ Connected to Home Assistant")
                
                # Start listening for messages
                asyncio.create_task(ha_listener())
                break
            else:
                logger.error(f"HA auth failed: {data}")
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"HA connection error: {e}")
            await asyncio.sleep(10)

async def ha_listener():
    """Listen for HA messages"""
    global ha_socket, ha_connected
    try:
        async for message in ha_socket:
            data = json.loads(message)
            # Handle any responses if needed
    except Exception as e:
        logger.error(f"HA listener error: {e}")
        ha_connected = False
        asyncio.create_task(connect_ha())

async def call_ha_service(service, entity_id):
    """Call Home Assistant service"""
    global ha_socket, ha_connected
    
    if not ha_connected or not ha_socket:
        logger.warning("HA not connected")
        return False
    
    try:
        message = {
            "id": int(time.time() * 1000),
            "type": "call_service",
            "domain": "light",
            "service": service,
            "service_data": {
                "entity_id": entity_id
            }
        }
        await ha_socket.send(json.dumps(message))
        logger.info(f"✅ {service} → {entity_id}")
        return True
    except Exception as e:
        logger.error(f"HA call error: {e}")
        ha_connected = False
        return False

async def trigger_action(finger_count):
    """Trigger HA action based on finger count"""
    global last_gesture, last_trigger_time, last_action
    
    now = time.time()
    
    # Cooldown check
    if last_gesture == finger_count and (now - last_trigger_time) < COOLDOWN_SECONDS:
        return None
    
    action = None
    entity = None
    service = None
    action_description = None
    
    if finger_count == 0:  # Fist
        action = "turn_on"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_on"
        action_description = "ON Strip Light"
    elif finger_count == 2:  # Peace
        action = "turn_on"
        entity = ROW_3_ENTITY
        service = "turn_on"
        action_description = "ON Row 3"
    elif finger_count == 3:  # Three
        action = "turn_off"
        entity = ROW_3_ENTITY
        service = "turn_off"
        action_description = "OFF Row 3"
    elif finger_count == 5:  # Palm
        action = "turn_off"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_off"
        action_description = "OFF Strip Light"
    
    if action and entity:
        last_gesture = finger_count
        last_trigger_time = now
        last_action = action_description
        
        logger.info(f"🎯 Triggering: {action_description}")
        success = await call_ha_service(service, entity)
        
        if success:
            # Broadcast action to all clients
            await sio.emit('action_triggered', {
                'action': action_description,
                'fingerCount': finger_count
            })
            return {"action": action, "entity": entity}
    
    return None

# ============================================================================
# CAMERA PROCESSING THREAD
# ============================================================================

def camera_processor():
    """Background thread for RTSP camera processing"""
    global current_frame, current_finger_count, current_gesture_name, current_gesture_icon
    global detection_status, camera_active, last_trigger_time, last_gesture
    
    if not RTSP_URL:
        logger.error("❌ No RTSP URL configured")
        return
    
    logger.info("📹 Starting camera processor")
    logger.info(f"   Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    logger.info(f"   FPS: {FPS}")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )
    
    cap = None
    reconnect_delay = 1
    frame_skip = max(1, int(30 / FPS))  # Skip frames to achieve target FPS
    frame_counter = 0
    
    # Get event loop for async operations
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                logger.info("🔌 Connecting to RTSP stream...")
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                
                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")
                
                logger.info("✅ RTSP stream connected")
                camera_active = True
                reconnect_delay = 1
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            frame_counter += 1
            
            # Process every frame for smooth video
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            detection_frame = frame.copy()
            
            if results.multi_hand_landmarks:
                detection_status = "detecting"
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Count fingers
                finger_count = count_extended_fingers(hand_landmarks)
                gesture = get_gesture_info(finger_count)
                
                # Update global state
                if current_finger_count != finger_count:
                    logger.debug(f"Gesture: {finger_count} fingers ({gesture['name']})")
                
                current_finger_count = finger_count
                current_gesture_name = gesture['name']
                current_gesture_icon = gesture['icon']
                
                # Draw landmarks on frame
                landmarks_list = []
                for i in range(21):
                    landmarks_list.append(hand_landmarks.landmark[i])
                detection_frame = draw_hand_landmarks(detection_frame, landmarks_list, finger_count)
                
                # Trigger action (async)
                future = asyncio.run_coroutine_threadsafe(
                    trigger_action(finger_count), 
                    loop
                )
                future.add_done_callback(lambda f: None)
            else:
                detection_status = "no_hand"
                current_finger_count = 0
                current_gesture_name = "No Hand"
                current_gesture_icon = "🤚"
                # Draw status on frame
                cv2.putText(detection_frame, "No Hand Detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add gesture name to frame
            cv2.putText(detection_frame, f"Gesture: {current_gesture_name}", (10, FRAME_HEIGHT - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update current frame
            with frame_lock:
                current_frame = detection_frame.copy()
            
            # Reset reconnect delay
            reconnect_delay = 1
            
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
            camera_active = False
            if cap:
                cap.release()
                cap = None
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

# ============================================================================
# VIDEO STREAM HANDLER
# ============================================================================

async def video_stream(request):
    """MJPEG video stream endpoint"""
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)
    
    while True:
        try:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    await asyncio.sleep(0.05)
                    continue
            
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_data = jpeg.tobytes()
            
            # Write frame to stream
            await response.write(b'--frame\r\n')
            await response.write(b'Content-Type: image/jpeg\r\n')
            await response.write(f'Content-Length: {len(frame_data)}\r\n\r\n'.encode())
            await response.write(frame_data)
            await response.write(b'\r\n')
            
            # Control frame rate
            await asyncio.sleep(1 / FPS)
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            break
    
    return response

# ============================================================================
# WEB SOCKET EVENTS
# ============================================================================

@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"🟢 Frontend connected: {sid}")
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cameraActive': camera_active,
        'cooldownRemaining': max(0, COOLDOWN_SECONDS - (time.time() - last_trigger_time)) if last_trigger_time else 0
    }, room=sid)

@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f"🔴 Frontend disconnected: {sid}")

async def broadcast_state():
    """Broadcast current state to all clients"""
    while True:
        try:
            cooldown = 0
            if last_trigger_time:
                remaining = COOLDOWN_SECONDS - (time.time() - last_trigger_time)
                cooldown = max(0, remaining)
            
            await sio.emit('gesture_update', {
                'fingerCount': current_finger_count,
                'gestureName': current_gesture_name,
                'gestureIcon': current_gesture_icon,
                'lastAction': last_action,
                'status': detection_status,
                'cameraActive': camera_active,
                'cooldownRemaining': cooldown
            })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.1)

# ============================================================================
# HTTP ROUTES
# ============================================================================

async def index_handler(request):
    """Serve the frontend UI"""
    html_path = '/app/web/index.html'
    try:
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = get_embedded_html()
        
        return web.Response(text=html_content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return web.Response(text=f"Error: {e}", status=500)

def get_embedded_html():
    """Return embedded HTML"""
    return """<!DOCTYPE html>
<html>
<head><title>Gesture Control</title></head>
<body>
    <h1>Gesture Control Add-on</h1>
    <p>Please ensure web/index.html exists</p>
</body>
</html>"""

async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'running',
        'rtsp_configured': bool(RTSP_URL),
        'camera_active': camera_active,
        'gesture': current_gesture_name,
        'finger_count': current_finger_count,
        'ha_connected': ha_connected
    })

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Main async entry point"""
    logger.info("=" * 60)
    logger.info("🎮 Gesture Control Add-on with RTSP Camera")
    logger.info("=" * 60)
    logger.info(f"📹 RTSP URL: {'✅ Configured' if RTSP_URL else '❌ NOT CONFIGURED'}")
    logger.info(f"🔌 HA URL: {HA_URL}")
    logger.info(f"💡 Strip Light: {STRIP_LIGHT_ENTITY}")
    logger.info(f"💡 Row 3: {ROW_3_ENTITY}")
    logger.info(f"⏱️  Cooldown: {COOLDOWN_SECONDS}s")
    logger.info(f"📐 Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    logger.info(f"🎯 FPS: {FPS}")
    logger.info("=" * 60)
    
    # Connect to Home Assistant
    if HA_TOKEN:
        asyncio.create_task(connect_ha())
    else:
        logger.warning("⚠️ No HA token configured")
    
    # Start camera thread
    if RTSP_URL:
        camera_thread = threading.Thread(target=camera_processor, daemon=True)
        camera_thread.start()
        logger.info("✅ Camera thread started")
    else:
        logger.warning("⚠️ No RTSP URL configured")
    
    # Start broadcast task
    asyncio.create_task(broadcast_state())
    
    # Setup routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/video', video_stream)
    app.router.add_get('/health', health_handler)
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8099)
    await site.start()
    
    logger.info("=" * 60)
    logger.info("🌐 Web server started on port 8099")
    logger.info(f"📍 Access the UI at: http://[YOUR-IP]:8099/")
    logger.info(f"🎥 Video stream at: http://[YOUR-IP]:8099/video")
    logger.info("=" * 60)
    
    await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)