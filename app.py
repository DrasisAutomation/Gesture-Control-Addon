#!/usr/bin/env python3
"""
Gesture Control Add-on with RTSP Camera and H.264 Live Video Stream
- RTSP camera stream processing with MediaPipe Hands
- H.264 video streaming with hardware acceleration
- WebSocket for real-time gesture updates
- Optimized buffer management
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
import websockets
from collections import deque
import io
import subprocess
import tempfile
import shutil
from fractions import Fraction

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
VIDEO_BITRATE = 1000  # kbps
USE_HARDWARE_ACCEL = True  # Use GPU/VAAPI if available

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
            VIDEO_BITRATE = config.get('video_bitrate', VIDEO_BITRATE)
            USE_HARDWARE_ACCEL = config.get('use_hardware_accel', USE_HARDWARE_ACCEL)
            logger.info("✅ Loaded configuration")
except Exception as e:
    logger.error(f"Error loading config: {e}")

# Global state
current_frame = None
current_frame_ready = False
current_finger_count = 0
current_gesture_name = "No Hand"
current_gesture_icon = "🤚"
last_action = ""
detection_status = "no_hand"
camera_active = False
last_trigger_time = 0
last_gesture = None
frame_lock = threading.Lock()
frame_counter = 0
last_frame_time = time.time()

# H.264 encoder process
encoder_process = None
encoder_lock = threading.Lock()
encoder_ready = False

# WebSocket server for frontend
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# HA WebSocket connection
ha_websocket = None
ha_connected = False

# ============================================================================
# H.264 ENCODER SETUP
# ============================================================================

def check_hardware_accel():
    """Check available hardware acceleration"""
    accel_options = []
    
    # Check for VAAPI (Intel/AMD)
    if shutil.which('vainfo'):
        try:
            result = subprocess.run(['vainfo'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                accel_options.append('vaapi')
        except:
            pass
    
    # Check for NVENC (NVIDIA)
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                accel_options.append('nvenc')
        except:
            pass
    
    return accel_options

def get_encoder_params():
    """Get optimal encoder parameters for H.264"""
    width = FRAME_WIDTH
    height = FRAME_HEIGHT
    fps = FPS
    bitrate = VIDEO_BITRATE * 1000  # Convert to bps
    
    # Ensure dimensions are even for encoder
    width = width - (width % 2)
    height = height - (height % 2)
    
    base_params = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', f'{bitrate}',
        '-pix_fmt', 'yuv420p',
        '-g', str(fps * 2),
        '-keyint_min', str(fps),
        '-sc_threshold', '0',
        '-f', 'mp4',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-pipe:1'
    ]
    
    # Try hardware acceleration if enabled
    if USE_HARDWARE_ACCEL:
        accel_options = check_hardware_accel()
        
        if 'vaapi' in accel_options:
            logger.info("🎬 Using VAAPI hardware acceleration")
            base_params = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', 'pipe:0',
                '-c:v', 'h264_vaapi',
                '-vf', 'format=nv12,hwupload',
                '-b:v', f'{bitrate}',
                '-global_quality', '25',
                '-f', 'mp4',
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-pipe:1'
            ]
        elif 'nvenc' in accel_options:
            logger.info("🎬 Using NVENC hardware acceleration")
            base_params = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', 'pipe:0',
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-tune', 'll',
                '-b:v', f'{bitrate}',
                '-rc', 'vbr',
                '-f', 'mp4',
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-pipe:1'
            ]
        else:
            logger.info("🎬 Using software encoding (libx264)")
    else:
        logger.info("🎬 Using software encoding (libx264)")
    
    return base_params

def start_encoder():
    """Start H.264 encoder process"""
    global encoder_process, encoder_ready
    
    with encoder_lock:
        if encoder_process is not None:
            try:
                encoder_process.terminate()
                encoder_process.wait(timeout=2)
            except:
                encoder_process.kill()
            encoder_process = None
        
        encoder_params = get_encoder_params()
        logger.info(f"Starting H.264 encoder: {' '.join(encoder_params)}")
        
        try:
            encoder_process = subprocess.Popen(
                encoder_params,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            encoder_ready = True
            logger.info("✅ H.264 encoder started")
            return True
        except Exception as e:
            logger.error(f"Failed to start encoder: {e}")
            encoder_ready = False
            return False

def encode_frame_to_h264(frame):
    """Encode a single frame to H.264 format"""
    global encoder_process, encoder_ready
    
    if not encoder_ready or encoder_process is None:
        if not start_encoder():
            return None
    
    try:
        # Ensure frame is continuous in memory
        frame_data = np.ascontiguousarray(frame).tobytes()
        
        # Write to encoder stdin
        encoder_process.stdin.write(frame_data)
        encoder_process.stdin.flush()
        
        # Read encoded data
        # Read until we have a complete frame (check for moov atoms)
        encoded_data = b''
        timeout_start = time.time()
        
        while time.time() - timeout_start < 0.1:  # 100ms timeout
            # Read available data
            data = encoder_process.stdout.read(4096)
            if not data:
                break
            encoded_data += data
            
            # Check if we have a complete MP4 fragment
            if b'moof' in encoded_data and len(encoded_data) > 1000:
                break
        
        if encoded_data:
            return encoded_data
        return None
        
    except (BrokenPipeError, OSError) as e:
        logger.error(f"Encoder pipe error: {e}")
        encoder_ready = False
        encoder_process = None
        return None
    except Exception as e:
        logger.error(f"Frame encoding error: {e}")
        return None

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
        [0,1], [1,2], [2,3], [3,4],
        [0,5], [5,6], [6,7], [7,8],
        [0,9], [9,10], [10,11], [11,12],
        [0,13], [13,14], [14,15], [15,16],
        [0,17], [17,18], [18,19], [19,20],
        [5,9], [9,13], [13,17]
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
    
    return frame

# ============================================================================
# HOME ASSISTANT CONNECTION
# ============================================================================

async def connect_ha():
    """Connect to Home Assistant WebSocket"""
    global ha_websocket, ha_connected
    
    while True:
        try:
            logger.info(f"🔌 Connecting to Home Assistant at {HA_URL}")
            ha_websocket = await websockets.connect(HA_URL)
            
            response = await ha_websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "auth_required":
                auth_msg = {
                    "type": "auth",
                    "access_token": HA_TOKEN
                }
                await ha_websocket.send(json.dumps(auth_msg))
                
                response = await ha_websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "auth_ok":
                    ha_connected = True
                    logger.info("✅ Connected to Home Assistant")
                    asyncio.create_task(ha_listener())
                    break
                else:
                    logger.error(f"HA auth failed: {data}")
            else:
                logger.error(f"Unexpected HA message: {data}")
                
        except Exception as e:
            logger.error(f"HA connection error: {e}")
            ha_connected = False
        
        await asyncio.sleep(10)

async def ha_listener():
    """Listen for HA messages"""
    global ha_websocket, ha_connected
    try:
        async for message in ha_websocket:
            data = json.loads(message)
            if data.get("type") == "pong":
                pass
    except Exception as e:
        logger.error(f"HA listener error: {e}")
        ha_connected = False
        asyncio.create_task(connect_ha())

async def call_ha_service(service, entity_id):
    """Call Home Assistant service"""
    global ha_websocket, ha_connected
    
    if not ha_connected or not ha_websocket:
        return False
    
    try:
        msg_id = int(time.time() * 1000)
        message = {
            "id": msg_id,
            "type": "call_service",
            "domain": "light",
            "service": service,
            "service_data": {"entity_id": entity_id}
        }
        
        await ha_websocket.send(json.dumps(message))
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
    
    if last_gesture == finger_count and (now - last_trigger_time) < COOLDOWN_SECONDS:
        return None
    
    action = None
    entity = None
    service = None
    action_description = None
    
    if finger_count == 0:
        action = "turn_on"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_on"
        action_description = "ON Strip Light"
    elif finger_count == 2:
        action = "turn_on"
        entity = ROW_3_ENTITY
        service = "turn_on"
        action_description = "ON Row 3"
    elif finger_count == 3:
        action = "turn_off"
        entity = ROW_3_ENTITY
        service = "turn_off"
        action_description = "OFF Row 3"
    elif finger_count == 5:
        action = "turn_off"
        entity = STRIP_LIGHT_ENTITY
        service = "turn_off"
        action_description = "OFF Strip Light"
    
    if action and entity:
        last_gesture = finger_count
        last_trigger_time = now
        last_action = action_description
        
        logger.info(f"🎯 {action_description}")
        success = await call_ha_service(service, entity)
        
        if success:
            await sio.emit('action_triggered', {
                'action': action_description,
                'fingerCount': finger_count
            })
            return {"action": action, "entity": entity}
    
    return None

# ============================================================================
# IMPROVED CAMERA PROCESSING WITH ERROR HANDLING
# ============================================================================

def camera_processor():
    """Background thread for RTSP camera processing with H.264 encoding"""
    global current_frame, current_frame_ready, current_finger_count, current_gesture_name
    global current_gesture_icon, detection_status, camera_active, frame_counter, last_frame_time
    global encoder_ready
    
    if not RTSP_URL:
        logger.error("❌ No RTSP URL configured")
        return
    
    logger.info("📹 Starting camera processor with H.264 encoding")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,  # Use lighter model for better performance
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )
    
    cap = None
    reconnect_delay = 1
    stable_frame_count = 0
    last_valid_frame = None
    
    # Start H.264 encoder
    start_encoder()
    
    # Create a blank frame for when camera is disconnected
    blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Connecting to camera...", (50, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
                
                # Show connecting message
                with frame_lock:
                    current_frame = blank_frame.copy()
                    current_frame_ready = True
                
                # Reinitialize encoder on reconnect
                if not encoder_ready:
                    start_encoder()
                
                cap = cv2.VideoCapture(RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                
                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")
                
                logger.info("✅ RTSP stream connected")
                camera_active = True
                reconnect_delay = 1
                stable_frame_count = 0
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            stable_frame_count += 1
            
            # Skip first few frames to let stream stabilize
            if stable_frame_count < 5:
                time.sleep(0.05)
                continue
            
            # Resize and process
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            # Create output frame
            output_frame = frame.copy()
            
            if results.multi_hand_landmarks:
                detection_status = "detecting"
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Count fingers
                finger_count = count_extended_fingers(hand_landmarks)
                gesture = get_gesture_info(finger_count)
                
                if current_finger_count != finger_count and finger_count > 0:
                    logger.info(f"🖐️ Gesture: {finger_count} fingers ({gesture['name']})")
                
                current_finger_count = finger_count
                current_gesture_name = gesture['name']
                current_gesture_icon = gesture['icon']
                
                # Draw landmarks
                landmarks_list = []
                for i in range(21):
                    landmarks_list.append(hand_landmarks.landmark[i])
                output_frame = draw_hand_landmarks(output_frame, landmarks_list, finger_count)
                
                # Trigger action
                future = asyncio.run_coroutine_threadsafe(
                    trigger_action(finger_count), loop
                )
                future.add_done_callback(lambda f: None)
            else:
                detection_status = "no_hand"
                current_finger_count = 0
                current_gesture_name = "No Hand"
                current_gesture_icon = "🤚"
                cv2.putText(output_frame, "No Hand Detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add overlay information
            # Gesture name
            cv2.putText(output_frame, f"Gesture: {current_gesture_name}", (10, FRAME_HEIGHT - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Finger count
            cv2.putText(output_frame, f"Fingers: {current_finger_count}", (10, FRAME_HEIGHT - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # HA Status
            ha_text = "HA: Connected" if ha_connected else "HA: Disconnected"
            ha_color = (0, 255, 0) if ha_connected else (0, 0, 255)
            cv2.putText(output_frame, ha_text, (FRAME_WIDTH - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ha_color, 1)
            
            # Encoder status
            encoder_text = "H.264: Active" if encoder_ready else "H.264: Inactive"
            cv2.putText(output_frame, encoder_text, (FRAME_WIDTH - 150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Cooldown
            if last_trigger_time > 0:
                remaining = COOLDOWN_SECONDS - (time.time() - last_trigger_time)
                if remaining > 0:
                    cv2.putText(output_frame, f"Cooldown: {remaining:.1f}s", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Update frame counter for FPS
            frame_counter += 1
            now = time.time()
            if now - last_frame_time >= 1.0:
                fps = frame_counter / (now - last_frame_time)
                cv2.putText(output_frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 80, FRAME_HEIGHT - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                frame_counter = 0
                last_frame_time = now
            
            # Store the processed frame
            with frame_lock:
                current_frame = output_frame.copy()
                current_frame_ready = True
            
            # Control frame rate
            time.sleep(1.0 / FPS)
            
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
            camera_active = False
            
            # Show error frame
            error_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Camera Error: Reconnecting in {reconnect_delay}s", 
                       (10, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            with frame_lock:
                current_frame = error_frame.copy()
                current_frame_ready = True
            
            if cap:
                cap.release()
                cap = None
            
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

# ============================================================================
# H.264 VIDEO STREAM HANDLER
# ============================================================================

async def video_stream(request):
    """H.264 video stream with MP4 fragmentation"""
    response = web.StreamResponse()
    response.headers.update({
        'Content-Type': 'video/mp4',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Accept-Ranges': 'none'
    })
    await response.prepare(request)
    
    last_frame_sent = None
    frame_count = 0
    
    try:
        while True:
            with frame_lock:
                if current_frame_ready and current_frame is not None:
                    frame = current_frame.copy()
                else:
                    await asyncio.sleep(0.05)
                    continue
            
            # Encode frame to H.264
            encoded_data = await asyncio.get_event_loop().run_in_executor(
                None, encode_frame_to_h264, frame
            )
            
            if encoded_data:
                # Send H.264/MP4 fragment
                await response.write(encoded_data)
                frame_count += 1
            
            # Control frame rate
            await asyncio.sleep(1.0 / FPS)
            
    except Exception as e:
        logger.error(f"H.264 stream error: {e}")
    
    return response

# ============================================================================
# ALTERNATIVE MJPEG STREAM (Fallback)
# ============================================================================

async def mjpeg_stream(request):
    """MJPEG video stream as fallback"""
    response = web.StreamResponse()
    response.headers.update({
        'Content-Type': 'multipart/x-mixed-replace; boundary=--jpgboundary',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    await response.prepare(request)
    
    try:
        while True:
            with frame_lock:
                if current_frame_ready and current_frame is not None:
                    frame = current_frame.copy()
                else:
                    await asyncio.sleep(0.05)
                    continue
            
            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, jpeg = cv2.imencode('.jpg', frame, encode_param)
            frame_data = jpeg.tobytes()
            
            # Send frame
            await response.write(b'--jpgboundary\r\n')
            await response.write(b'Content-Type: image/jpeg\r\n')
            await response.write(f'Content-Length: {len(frame_data)}\r\n\r\n'.encode())
            await response.write(frame_data)
            await response.write(b'\r\n')
            
            await asyncio.sleep(1.0 / FPS)
            
    except Exception as e:
        logger.error(f"MJPEG stream error: {e}")
    
    return response

# ============================================================================
# WEB SOCKET EVENTS
# ============================================================================

@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"🟢 Frontend connected: {sid}")
    cooldown = max(0, COOLDOWN_SECONDS - (time.time() - last_trigger_time)) if last_trigger_time else 0
    await sio.emit('gesture_update', {
        'fingerCount': current_finger_count,
        'gestureName': current_gesture_name,
        'gestureIcon': current_gesture_icon,
        'lastAction': last_action,
        'status': detection_status,
        'cameraActive': camera_active,
        'haConnected': ha_connected,
        'cooldownRemaining': cooldown,
        'encoderActive': encoder_ready
    }, room=sid)

@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f"🔴 Frontend disconnected: {sid}")

async def broadcast_state():
    """Broadcast current state to all clients"""
    while True:
        try:
            cooldown = max(0, COOLDOWN_SECONDS - (time.time() - last_trigger_time)) if last_trigger_time else 0
            
            await sio.emit('gesture_update', {
                'fingerCount': current_finger_count,
                'gestureName': current_gesture_name,
                'gestureIcon': current_gesture_icon,
                'lastAction': last_action,
                'status': detection_status,
                'cameraActive': camera_active,
                'haConnected': ha_connected,
                'cooldownRemaining': cooldown,
                'encoderActive': encoder_ready
            })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.1)

# ============================================================================
# HTTP ROUTES
# ============================================================================

async def index_handler(request):
    """Serve the frontend UI"""
    html_paths = [
        '/app/web/index.html',
        './web/index.html',
        os.path.join(os.path.dirname(__file__), 'web', 'index.html')
    ]
    
    html_content = None
    for path in html_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            break
    
    if not html_content:
        html_content = """<!DOCTYPE html>
<html><head><title>Gesture Control</title></head>
<body><h1>Gesture Control Add-on</h1><p>Loading...</p></body></html>"""
    
    return web.Response(text=html_content, content_type='text/html')

async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'running',
        'rtsp_configured': bool(RTSP_URL),
        'camera_active': camera_active,
        'gesture': current_gesture_name,
        'finger_count': current_finger_count,
        'ha_connected': ha_connected,
        'encoder_active': encoder_ready,
        'video_format': 'h264'
    })

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def cleanup_encoder():
    """Clean up encoder process on shutdown"""
    global encoder_process
    if encoder_process:
        try:
            encoder_process.terminate()
            encoder_process.wait(timeout=2)
        except:
            encoder_process.kill()
        logger.info("Encoder process terminated")

async def main():
    """Main async entry point"""
    logger.info("=" * 60)
    logger.info("🎮 Gesture Control Add-on with H.264 Camera Stream")
    logger.info("=" * 60)
    logger.info(f"📹 RTSP: {'✅' if RTSP_URL else '❌'}")
    logger.info(f"🎬 H.264 Encoding: {'✅' if USE_HARDWARE_ACCEL else '⚠️ Software'}")
    logger.info(f"🎬 Bitrate: {VIDEO_BITRATE} kbps")
    logger.info(f"🔌 HA: {HA_URL}")
    logger.info(f"💡 Strip: {STRIP_LIGHT_ENTITY}")
    logger.info(f"💡 Row 3: {ROW_3_ENTITY}")
    logger.info("=" * 60)
    
    # Check hardware acceleration
    if USE_HARDWARE_ACCEL:
        accel = check_hardware_accel()
        if accel:
            logger.info(f"✅ Hardware acceleration available: {', '.join(accel)}")
        else:
            logger.warning("⚠️ No hardware acceleration found, using software encoding")
    
    # Connect to HA
    if HA_TOKEN:
        asyncio.create_task(connect_ha())
    
    # Start camera thread
    if RTSP_URL:
        camera_thread = threading.Thread(target=camera_processor, daemon=True)
        camera_thread.start()
        logger.info("✅ Camera thread started")
    
    # Start broadcast
    asyncio.create_task(broadcast_state())
    
    # Setup routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/video', video_stream)
    app.router.add_get('/mjpeg', mjpeg_stream)  # Fallback MJPEG stream
    app.router.add_get('/health', health_handler)
    
    # Register cleanup
    app.on_shutdown.append(lambda _: cleanup_encoder())
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8099)
    await site.start()
    
    logger.info("=" * 60)
    logger.info("🌐 Server: http://0.0.0.0:8099")
    logger.info("🎥 H.264 Stream: http://[IP]:8099/video")
    logger.info("🎥 MJPEG Stream (fallback): http://[IP]:8099/mjpeg")
    logger.info("=" * 60)
    
    await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)