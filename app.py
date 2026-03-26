#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
- RTSP camera stream processing with MediaPipe Hands
- WebSocket server for frontend UI updates
- MJPEG stream for camera feed
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
import os
import sys
import base64
import httpx
import socketio
from aiohttp import web
from aiohttp import web_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
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
            logger.info(f"   RTSP URL: {RTSP_URL[:50] if RTSP_URL else 'NOT SET'}...")
            logger.info(f"   Strip Light: {STRIP_LIGHT_ENTITY}")
            logger.info(f"   Row 3: {ROW_3_ENTITY}")
            logger.info(f"   Cooldown: {COOLDOWN_SECONDS}s")
    else:
        logger.warning(f"Config file not found: {config_path}")
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

# Frame storage for MJPEG stream
current_frame = None
frame_lock = threading.Lock()

# WebSocket server for frontend
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
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


def draw_hand_landmarks(frame, landmarks):
    """Draw hand landmarks on frame for visual feedback"""
    h, w, _ = frame.shape
    
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
        x1, y1 = int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h)
        x2, y2 = int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw landmarks
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        is_tip = i in [4, 8, 12, 16, 20]
        color = (0, 255, 255) if is_tip else (0, 0, 255)
        radius = 6 if is_tip else 4
        cv2.circle(frame, (x, y), radius, color, -1)
    
    return frame


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
        last_action = f"ON {entity.split('.')[1]}"
    elif finger_count == 2:
        action = "turn_on"
        entity = ROW_3_ENTITY
        service = "turn_on"
        last_action = f"ON {entity.split('.')[1]}"
    elif finger_count == 3:
        action = "turn_off"
        entity = ROW_3_ENTITY
        service = "turn_off"
        last_action = f"OFF {entity.split('.')[1]}"
    elif finger_count == 5:
        action = "turn_off"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_off"
        last_action = f"OFF {entity.split('.')[1]}"
    
    if action and entity:
        logger.info(f"🎯 Gesture detected: {get_gesture_info(finger_count)['name']} → {last_action}")
        await call_ha_service(service, entity)
        return {"action": action, "entity": entity}
    
    return None


# -------------------------------------------------------------------------
# RTSP Camera Processing Thread
# -------------------------------------------------------------------------

def camera_processor():
    """Background thread for RTSP camera processing"""
    global current_finger_count, current_gesture_name, current_gesture_icon, detection_status, current_frame
    
    if not RTSP_URL:
        logger.error("❌ No RTSP URL configured, camera processor stopped")
        return
    
    logger.info(f"📹 Starting camera processor with URL: {RTSP_URL[:50]}...")
    
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
    frame_count = 0
    last_log_time = 0
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                logger.info(f"🔌 Connecting to RTSP stream...")
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")
                logger.info("✅ RTSP stream connected successfully")
                reconnect_delay = 1
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            frame_count += 1
            # Process every frame for better detection
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            # Draw debug info
            if results.multi_hand_landmarks:
                detection_status = "detecting"
                landmarks = results.multi_hand_landmarks[0]
                
                # Count fingers
                finger_count = count_extended_fingers(landmarks)
                gesture = get_gesture_info(finger_count)
                
                # Update global state
                if current_finger_count != finger_count:
                    logger.info(f"🖐️ Gesture changed: {finger_count} fingers → {gesture['name']}")
                
                current_finger_count = finger_count
                current_gesture_name = gesture['name']
                current_gesture_icon = gesture['icon']
                
                # Draw hand landmarks on frame
                frame = draw_hand_landmarks(frame, landmarks)
                
                # Add text overlay
                cv2.putText(frame, f"Gesture: {gesture['name']}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show cooldown if active
                now = time.time()
                if last_trigger_time and (now - last_trigger_time) < COOLDOWN_SECONDS:
                    remaining = COOLDOWN_SECONDS - (now - last_trigger_time)
                    cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Trigger HA action
                try:
                    loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        trigger_action(finger_count), loop
                    )
                except Exception as e:
                    logger.error(f"Error triggering action: {e}")
            else:
                if detection_status != "no_hand":
                    logger.debug("No hand detected")
                detection_status = "no_hand"
                current_finger_count = 0
                current_gesture_name = "No Hand"
                current_gesture_icon = "🤚"
                
                cv2.putText(frame, "No hand detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Store frame for MJPEG stream
            with frame_lock:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                current_frame = jpeg.tobytes()
            
            # Log detection every 30 seconds
            if time.time() - last_log_time > 30:
                if detection_status == "detecting":
                    logger.info(f"📊 Active detection: {current_gesture_name} ({current_finger_count} fingers)")
                last_log_time = time.time()
            
            # Reset reconnect delay on successful frame
            reconnect_delay = 1
            
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
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
    logger.info(f"🟢 Frontend connected: {sid}")
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cooldownRemaining': max(0, COOLDOWN_SECONDS - (time.time() - last_trigger_time)) if last_trigger_time else 0
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
        'cooldownRemaining': cooldown_remaining
    })


# -------------------------------------------------------------------------
# HTTP Routes
# -------------------------------------------------------------------------

async def index_handler(request):
    """Serve the frontend UI"""
    try:
        possible_paths = [
            '/app/web/index.html',
            './web/index.html',
        ]
        
        html_content = None
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"📄 Found index.html at {path}")
                with open(path, 'r') as f:
                    html_content = f.read()
                break
        
        if html_content is None:
            logger.error("❌ index.html not found")
            return web.Response(text="Error: UI files not found", status=500)
        
        return web.Response(text=html_content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return web.Response(text=f"Error loading UI: {str(e)}", status=500)


async def video_feed_handler(request):
    """MJPEG stream endpoint"""
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    await response.prepare(request)
    
    logger.info("📹 MJPEG stream started")
    
    while True:
        try:
            if current_frame:
                await response.write(b'--frame\r\n')
                await response.write(b'Content-Type: image/jpeg\r\n')
                await response.write(f'Content-Length: {len(current_frame)}\r\n\r\n'.encode())
                await response.write(current_frame)
                await response.write(b'\r\n')
            await asyncio.sleep(0.033)  # ~30fps
        except Exception as e:
            logger.error(f"MJPEG stream error: {e}")
            break
    
    logger.info("📹 MJPEG stream ended")
    return response


async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'running',
        'rtsp_configured': bool(RTSP_URL),
        'rtsp_url': RTSP_URL[:50] if RTSP_URL else '',
        'gesture': current_gesture_name,
        'finger_count': current_finger_count,
        'detection_status': detection_status,
        'strip_light_entity': STRIP_LIGHT_ENTITY,
        'row_3_entity': ROW_3_ENTITY,
        'cooldown_seconds': COOLDOWN_SECONDS
    })


async def test_handler(request):
    """Test endpoint"""
    return web.Response(text="✅ Gesture Control Add-on is running!")


# -------------------------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------------------------

@web.middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses"""
    if request.method == 'OPTIONS':
        return web.Response(headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        })
    
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


# -------------------------------------------------------------------------
# Background Task for Broadcasting
# -------------------------------------------------------------------------

async def broadcast_task():
    """Periodically broadcast gesture state to frontend"""
    while True:
        try:
            await broadcast_gesture_state()
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.1)


# -------------------------------------------------------------------------
# Main Application Setup
# -------------------------------------------------------------------------

async def main():
    """Main async entry point"""
    logger.info("=" * 60)
    logger.info("🎮 Starting Gesture Control Add-on")
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
        logger.warning("   Please configure rtsp_url in add-on options and restart.")
    
    # Start broadcast task
    asyncio.create_task(broadcast_task())
    
    # Add CORS middleware
    app.middlewares.append(cors_middleware)
    
    # Setup HTTP routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/video_feed', video_feed_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/test', test_handler)
    
    # Run web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8099)
    await site.start()
    logger.info("=" * 60)
    logger.info("🌐 Web server started on port 8099")
    logger.info(f"📍 Access the UI at: http://[YOUR-IP]:8099/")
    logger.info(f"📹 Camera feed: http://[YOUR-IP]:8099/video_feed")
    logger.info(f"🔍 Health check: http://[YOUR-IP]:8099/health")
    logger.info(f"🧪 Test endpoint: http://[YOUR-IP]:8099/test")
    logger.info("=" * 60)
    
    # Keep running
    await asyncio.Event().wait()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)