#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
- RTSP camera stream processing with MediaPipe Hands
- WebSocket server for frontend UI updates
- Home Assistant API integration via Supervisor
"""

import asyncio
import cv2
import numpy as np
import mediapipe as mp
import time
import json
import logging
import threading
import queue
import os
import signal
import sys
from datetime import datetime
import httpx
import socketio
from aiohttp import web
import aiohttp_cors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
RTSP_URL = os.environ.get('RTSP_URL', '')
STRIP_LIGHT_ENTITY = os.environ.get('STRIP_LIGHT_ENTITY', 'light.strip_light')
ROW_3_ENTITY = os.environ.get('ROW_3_ENTITY', 'light.row_3')
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 2))
DETECTION_CONFIDENCE = float(os.environ.get('DETECTION_CONFIDENCE', 0.5))
TRACKING_CONFIDENCE = float(os.environ.get('TRACKING_CONFIDENCE', 0.5))

# Try to load config from add-on options
try:
    with open('/data/options.json', 'r') as f:
        config = json.load(f)
        RTSP_URL = config.get('rtsp_url', RTSP_URL)
        STRIP_LIGHT_ENTITY = config.get('strip_light_entity', STRIP_LIGHT_ENTITY)
        ROW_3_ENTITY = config.get('row_3_entity', ROW_3_ENTITY)
        COOLDOWN_SECONDS = config.get('cooldown_seconds', COOLDOWN_SECONDS)
        DETECTION_CONFIDENCE = config.get('detection_confidence', DETECTION_CONFIDENCE)
        TRACKING_CONFIDENCE = config.get('tracking_confidence', TRACKING_CONFIDENCE)
        logger.info(f"Loaded config from /data/options.json")
except FileNotFoundError:
    logger.warning("No /data/options.json found, using environment variables")

if not RTSP_URL:
    logger.error("RTSP_URL not configured! Please set in add-on options.")
    # Don't exit, let it run with error state

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

# Thread-safe queue for frame processing
frame_queue = queue.Queue(maxsize=2)

# WebSocket server for frontend
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# -------------------------------------------------------------------------
# MediaPipe Hands Finger Counting (EXACT same logic as HTML version)
# -------------------------------------------------------------------------

def count_extended_fingers(landmarks):
    """Count extended fingers - exact logic from original HTML"""
    if not landmarks:
        return 0
    
    count = 0
    
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
    elif finger_count == 2:
        return {'name': '2 Fingers', 'icon': '✌️'}
    elif finger_count == 3:
        return {'name': '3 Fingers', 'icon': '👌'}
    elif finger_count == 5:
        return {'name': 'Palm', 'icon': '✋'}
    else:
        plural = '' if finger_count == 1 else 's'
        return {'name': f'{finger_count} Finger{plural}', 'icon': '🖐️'}


# -------------------------------------------------------------------------
# Home Assistant API Calls
# -------------------------------------------------------------------------

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
                logger.error(f"HA API error: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        logger.error(f"HA API exception: {e}")
        return False


async def trigger_action(finger_count):
    """Trigger HA action based on finger count (with cooldown)"""
    global last_gesture, last_trigger_time, last_action
    
    now = time.time()
    
    # Cooldown check
    if last_gesture == finger_count and (now - last_trigger_time) < COOLDOWN_SECONDS:
        return None
    
    last_gesture = finger_count
    last_trigger_time = now
    
    action = None
    entity = None
    service = None
    
    if finger_count == 0:
        action = "turn_on"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_on"
    elif finger_count == 2:
        action = "turn_on"
        entity = ROW_3_ENTITY
        service = "turn_on"
    elif finger_count == 3:
        action = "turn_off"
        entity = ROW_3_ENTITY
        service = "turn_off"
    elif finger_count == 5:
        action = "turn_off"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_off"
    
    if action and entity:
        last_action = f"{action.split('_')[1].upper()} {entity.split('.')[1]}"
        await call_ha_service(service, entity)
        return {"action": action, "entity": entity}
    
    return None


# -------------------------------------------------------------------------
# RTSP Camera Processing Thread
# -------------------------------------------------------------------------

def camera_processor():
    """Background thread for RTSP camera processing"""
    global current_finger_count, current_gesture_name, current_gesture_icon, detection_status
    
    if not RTSP_URL:
        logger.error("No RTSP URL configured, camera processor stopped")
        return
    
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
    frame_skip = 0
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                logger.info(f"Connecting to RTSP: {RTSP_URL[:50]}...")
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")
                reconnect_delay = 1
                logger.info("RTSP stream connected")
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            # Process every 2nd frame to reduce CPU usage
            frame_skip = (frame_skip + 1) % 2
            if frame_skip == 0:
                # Flip horizontally for mirror effect (like original)
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    detection_status = "detecting"
                    landmarks = results.multi_hand_landmarks[0]
                    
                    # Count fingers using exact logic
                    finger_count = count_extended_fingers(landmarks)
                    gesture = get_gesture_info(finger_count)
                    
                    current_finger_count = finger_count
                    current_gesture_name = gesture['name']
                    current_gesture_icon = gesture['icon']
                    
                    # Trigger HA action (async, but called from thread)
                    loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        trigger_action(finger_count), loop
                    )
                else:
                    detection_status = "no_hand"
                    current_finger_count = 0
                    current_gesture_name = "No Hand"
                    current_gesture_icon = "🤚"
            
            # Reset reconnect delay on successful frame
            reconnect_delay = 1
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
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
        time.sleep(0.033)  # ~30fps


# -------------------------------------------------------------------------
# WebSocket Events for Frontend
# -------------------------------------------------------------------------

@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"Frontend connected: {sid}")
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cooldownRemaining': max(0, COOLDOWN_SECONDS - (time.time() - last_trigger_time)) if last_trigger_time else 0
    }, room=sid)


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
        'cooldownRemaining': cooldown_remaining
    })


# -------------------------------------------------------------------------
# HTTP Routes
# -------------------------------------------------------------------------

async def index_handler(request):
    """Serve the frontend UI"""
    try:
        with open('/app/web/index.html', 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return web.Response(text="Error loading UI", status=500)


async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'running',
        'rtsp_configured': bool(RTSP_URL),
        'gesture': current_gesture_name,
        'finger_count': current_finger_count,
        'detection_status': detection_status
    })


# -------------------------------------------------------------------------
# Background Task for Broadcasting
# -------------------------------------------------------------------------

async def broadcast_task():
    """Periodically broadcast gesture state to frontend"""
    while True:
        await broadcast_gesture_state()
        await asyncio.sleep(0.1)  # 10fps updates


# -------------------------------------------------------------------------
# Main Application Setup
# -------------------------------------------------------------------------

async def main():
    """Main async entry point"""
    # Start camera processing thread if RTSP is configured
    if RTSP_URL:
        camera_thread = threading.Thread(target=camera_processor, daemon=True)
        camera_thread.start()
        logger.info("Camera processing thread started")
    else:
        logger.warning("No RTSP URL configured. Camera processing disabled. Set rtsp_url in add-on options.")
    
    # Start broadcast task
    asyncio.create_task(broadcast_task())
    
    # Setup HTTP routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Run web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8099)
    await site.start()
    logger.info("Web server started on port 8099")
    
    # Keep running
    await asyncio.Event().wait()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)