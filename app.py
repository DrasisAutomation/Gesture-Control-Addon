#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
- RTSP camera stream processing with MediaPipe Hands
- WebSocket server for frontend UI updates
- ACCURATE finger counting with proper MediaPipe landmark handling
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
from aiohttp import web
import socketio
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RTSP_URL = ""
STRIP_LIGHT_ENTITY = "light.strip_light"
ROW_3_ENTITY = "light.row_3"
COOLDOWN_SECONDS = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Try to load config from add-on options
try:
    config_path = "/data/options.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            RTSP_URL = config.get('rtsp_url', '')
            STRIP_LIGHT_ENTITY = config.get('strip_light_entity', STRIP_LIGHT_ENTITY)
            ROW_3_ENTITY = config.get('row_3_entity', ROW_3_ENTITY)
            COOLDOWN_SECONDS = config.get('cooldown_seconds', COOLDOWN_SECONDS)
            DETECTION_CONFIDENCE = config.get('detection_confidence', DETECTION_CONFIDENCE)
            TRACKING_CONFIDENCE = config.get('tracking_confidence', TRACKING_CONFIDENCE)
            logger.info(f"✅ Loaded config from /data/options.json")
            logger.info(f"   Strip Light: {STRIP_LIGHT_ENTITY}")
            logger.info(f"   Row 3: {ROW_3_ENTITY}")
            logger.info(f"   Cooldown: {COOLDOWN_SECONDS}s")
except Exception as e:
    logger.error(f"Error loading config: {e}")

SUPERVISOR_TOKEN = os.environ.get('SUPERVISOR_TOKEN', '')
SUPERVISOR_API = "http://supervisor/core"

# Gesture state
last_gesture = None
last_trigger_time = 0
current_finger_count = 0
current_gesture_name = "No Hand"
current_gesture_icon = "🤚"
last_action = ""
detection_status = "no_hand"
camera_active = False

# WebSocket server
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# ============================================================================
# ACCURATE FINGER COUNTING - FIXED
# ============================================================================

def count_extended_fingers(hand_landmarks):
    """
    Count extended fingers from MediaPipe landmarks
    hand_landmarks is a NormalizedLandmarkList object
    """
    if not hand_landmarks:
        return 0
    
    count = 0
    
    # Convert to list for easier access
    landmarks = []
    for i in range(21):  # MediaPipe has 21 landmarks
        landmarks.append(hand_landmarks.landmark[i])
    
    # Index finger (tip 8 vs pip 6)
    if landmarks[8].y < landmarks[6].y - 0.03:
        count += 1
    
    # Middle finger (tip 12 vs pip 10)
    if landmarks[12].y < landmarks[10].y - 0.03:
        count += 1
    
    # Ring finger (tip 16 vs pip 14)
    if landmarks[16].y < landmarks[14].y - 0.02:
        count += 1
    
    # Pinky (tip 20 vs pip 18)
    if landmarks[20].y < landmarks[18].y - 0.02:
        count += 1
    
    # Thumb (horizontal position based on hand orientation)
    # Check if hand is right or left based on wrist and thumb positions
    is_right_hand = landmarks[5].x < landmarks[17].x
    if is_right_hand:
        # Right hand: thumb tip should be to the left of thumb IP
        if landmarks[4].x < landmarks[3].x - 0.02:
            count += 1
    else:
        # Left hand: thumb tip should be to the right of thumb IP
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


# ============================================================================
# Home Assistant API Calls
# ============================================================================

async def call_ha_service(service, entity_id):
    """Call Home Assistant service via Supervisor API"""
    if not SUPERVISOR_TOKEN:
        logger.error("No SUPERVISOR_TOKEN available")
        return False
    
    url = f"{SUPERVISOR_API}/api/services/light/{service}"
    headers = {
        "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"entity_id": entity_id}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=5.0)
            if response.status_code == 200:
                logger.info(f"✅ {service} → {entity_id}")
                return True
            else:
                logger.error(f"HA API error: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"HA API exception: {e}")
        return False


async def trigger_action(finger_count):
    """Trigger HA action based on finger count with cooldown"""
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
    elif finger_count == 2:  # Peace sign
        action = "turn_on"
        entity = ROW_3_ENTITY
        service = "turn_on"
        action_description = "ON Row 3"
    elif finger_count == 3:  # Three fingers
        action = "turn_off"
        entity = ROW_3_ENTITY
        service = "turn_off"
        action_description = "OFF Row 3"
    elif finger_count == 5:  # Open palm
        action = "turn_off"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_off"
        action_description = "OFF Strip Light"
    
    if action and entity:
        last_gesture = finger_count
        last_trigger_time = now
        last_action = action_description
        
        success = await call_ha_service(service, entity)
        if success:
            logger.info(f"🎯 Action triggered: {last_action}")
            return {"action": action, "entity": entity, "description": action_description}
    
    return None


# ============================================================================
# RTSP Camera Processing Thread - FIXED
# ============================================================================

def camera_processor():
    """Background thread for RTSP camera processing"""
    global current_finger_count, current_gesture_name, current_gesture_icon, detection_status, camera_active
    
    if not RTSP_URL:
        logger.error("❌ No RTSP URL configured, camera processor stopped")
        return
    
    logger.info(f"📹 Starting camera processor with MediaPipe Hands")
    logger.info(f"   RTSP URL: {RTSP_URL[:80]}...")
    
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
    max_reconnect_delay = 30
    last_landmarks = None
    last_detection_time = 0
    
    # Get or create event loop for async operations
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                logger.info(f"🔌 Connecting to RTSP stream...")
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")
                
                logger.info("✅ RTSP stream connected successfully")
                camera_active = True
                reconnect_delay = 1
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            # Process every frame
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                detection_status = "detecting"
                last_detection_time = time.time()
                # Get the first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                last_landmarks = hand_landmarks
                
                # Count fingers - FIXED: pass the landmark object directly
                finger_count = count_extended_fingers(hand_landmarks)
                gesture = get_gesture_info(finger_count)
                
                # Update global state
                if current_finger_count != finger_count:
                    logger.debug(f"Gesture: {finger_count} fingers ({gesture['name']})")
                
                current_finger_count = finger_count
                current_gesture_name = gesture['name']
                current_gesture_icon = gesture['icon']
                
                # Trigger HA action
                future = asyncio.run_coroutine_threadsafe(
                    trigger_action(finger_count), 
                    loop
                )
                # Optionally wait for result without blocking
                future.add_done_callback(lambda f: None)
            else:
                # No hand detected
                if time.time() - last_detection_time > 0.5:
                    if detection_status != "no_hand":
                        logger.debug("No hand detected")
                    detection_status = "no_hand"
                    current_finger_count = 0
                    current_gesture_name = "No Hand"
                    current_gesture_icon = "🤚"
                    last_landmarks = None
            
            # Reset reconnect delay on successful frame
            reconnect_delay = 1
            
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
            camera_active = False
            if cap:
                cap.release()
                cap = None
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            detection_status = "no_hand"
            current_finger_count = 0
            current_gesture_name = "No Hand"
            current_gesture_icon = "🤚"
        
        # Small sleep to prevent CPU hogging
        time.sleep(0.01)


# ============================================================================
# WebSocket Events
# ============================================================================

@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"🟢 Frontend connected: {sid}")
    cooldown_remaining = 0
    if last_trigger_time:
        remaining = COOLDOWN_SECONDS - (time.time() - last_trigger_time)
        cooldown_remaining = max(0, remaining)
    
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cameraActive': camera_active,
        'cooldownRemaining': cooldown_remaining
    }, room=sid)


@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f"🔴 Frontend disconnected: {sid}")


async def broadcast_gesture_state():
    """Broadcast current gesture state to all connected clients"""
    cooldown_remaining = 0
    if last_trigger_time:
        remaining = COOLDOWN_SECONDS - (time.time() - last_trigger_time)
        cooldown_remaining = max(0, remaining)
    
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cameraActive': camera_active,
        'cooldownRemaining': cooldown_remaining
    })


# ============================================================================
# HTTP Routes
# ============================================================================

async def index_handler(request):
    """Serve the frontend UI"""
    try:
        possible_paths = [
            '/app/web/index.html',
            './web/index.html',
            os.path.join(os.path.dirname(__file__), 'web', 'index.html')
        ]
        
        html_content = None
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"📄 Found index.html at {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                break
        
        if html_content is None:
            # Serve embedded HTML if file not found
            html_content = get_embedded_html()
        
        return web.Response(text=html_content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return web.Response(text=f"Error: {str(e)}", status=500)


def get_embedded_html():
    """Return embedded HTML if file not found"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Control - Loading</title>
    <style>
        body { background: #000; color: #fff; display: flex; justify-content: center; align-items: center; height: 100vh; font-family: monospace; }
        .error { text-align: center; }
    </style>
</head>
<body>
    <div class="error">
        <h2>⚠️ UI File Not Found</h2>
        <p>Please ensure index.html is in /app/web/ directory</p>
    </div>
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
        'detection_status': detection_status,
        'strip_light_entity': STRIP_LIGHT_ENTITY,
        'row_3_entity': ROW_3_ENTITY,
        'cooldown_seconds': COOLDOWN_SECONDS,
        'last_action': last_action
    })


# ============================================================================
# CORS Middleware
# ============================================================================

@web.middleware
async def cors_middleware(request, handler):
    """Add CORS headers"""
    if request.method == 'OPTIONS':
        return web.Response(headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        })
    
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


# ============================================================================
# Background Task
# ============================================================================

async def broadcast_task():
    """Periodically broadcast gesture state"""
    while True:
        try:
            await broadcast_gesture_state()
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.1)


# ============================================================================
# Main Application
# ============================================================================

async def main():
    """Main async entry point"""
    logger.info("=" * 60)
    logger.info("🎮 Starting Gesture Control Add-on (ACCURATE DETECTION)")
    logger.info("=" * 60)
    logger.info(f"📹 RTSP URL: {'✅ Configured' if RTSP_URL else '❌ NOT CONFIGURED'}")
    logger.info(f"💡 Strip Light: {STRIP_LIGHT_ENTITY}")
    logger.info(f"💡 Row 3 Light: {ROW_3_ENTITY}")
    logger.info(f"⏱️  Cooldown: {COOLDOWN_SECONDS}s")
    logger.info(f"🎯 Detection Confidence: {DETECTION_CONFIDENCE}")
    logger.info(f"🎯 Tracking Confidence: {TRACKING_CONFIDENCE}")
    logger.info("=" * 60)
    
    # Start camera processing thread if RTSP is configured
    if RTSP_URL:
        camera_thread = threading.Thread(target=camera_processor, daemon=True)
        camera_thread.start()
        logger.info("✅ Camera processing thread started")
    else:
        logger.warning("⚠️  No RTSP URL configured. Camera processing disabled.")
    
    # Start broadcast task
    asyncio.create_task(broadcast_task())
    
    # Add CORS middleware
    app.middlewares.append(cors_middleware)
    
    # Setup HTTP routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_options('/', cors_middleware)
    
    # Run web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8099)
    await site.start()
    
    logger.info("=" * 60)
    logger.info("🌐 Web server started on port 8099")
    logger.info(f"📍 Access the UI at: http://[YOUR-IP]:8099/")
    logger.info(f"🔍 Health check: http://[YOUR-IP]:8099/health")
    logger.info("=" * 60)
    
    # Keep running
    await asyncio.Event().wait()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)