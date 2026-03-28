#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
Enhanced with accurate palm detection, angle filtering, and stable processing
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import math
from collections import deque, Counter
from datetime import datetime
from pathlib import Path

import aiohttp
import aiohttp.web
import cv2
import mediapipe as mp
import numpy as np
import socketio

# ========== CONFIGURATION ==========
CONFIG_PATH = "/data/options.json"

def load_config():
    """Load configuration from Home Assistant add-on options"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        logger.info("✅ Loaded config from /data/options.json")
    except FileNotFoundError:
        logger.warning("⚠️ Config file not found, using defaults")
        config = {
            "rtsp_url": "rtsp://your-camera-ip:554/stream",
            "ha_url": "ws://homeassistant.local:8123/api/websocket",
            "ha_token": "",
            "entity_1": "light.light_1",
            "entity_2": "light.light_2",
            "entity_3": "light.light_3",
            "entity_4": "light.light_4",
            "entity_5": "light.light_5",
            "fps": 15,
            "frame_width": 640,
            "frame_height": 480,
            "bitrate": "500k",
            "cooldown_seconds": 1,
            "detection_confidence": 0.5,
            "tracking_confidence": 0.5,
            "reset_gesture": 0,
            "stability_frames": 5,
            "angle_tolerance": 0.4,  # 40% tolerance for palm angle
        }
    
    defaults = {
        "fps": 15,
        "frame_width": 640,
        "frame_height": 480,
        "bitrate": "500k",
        "cooldown_seconds": 1,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "reset_gesture": 0,
        "stability_frames": 5,
        "angle_tolerance": 0.4,
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gesture-control")

config = load_config()

# ========== IMPROVED GESTURE DETECTION ==========
mp_hands = mp.solutions.hands
hands = None

def init_mediapipe():
    """Initialize MediaPipe Hands"""
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config["detection_confidence"],
        min_tracking_confidence=config["tracking_confidence"],
        model_complexity=0
    )
    logger.info("✅ MediaPipe initialized")

def calculate_hand_angle(landmarks):
    """
    Calculate the angle of the palm to determine if hand is facing upward.
    Uses the angle between wrist and middle finger base.
    Returns True if hand is facing upward within tolerance.
    """
    if not landmarks or len(landmarks) < 9:
        return True  # Default to True if can't determine
    
    # Get wrist (0) and middle finger base (9) or index finger base (5)
    wrist = landmarks[0]
    middle_base = landmarks[9]  # Middle finger base
    
    # Calculate the vector from wrist to middle finger base
    dx = middle_base.x - wrist.x
    dy = middle_base.y - wrist.y
    
    # Normalize
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return True
    
    dx /= length
    dy /= length
    
    # For upward-facing palm, the hand should be vertical
    # dy should be negative (hand going up from wrist)
    # Allow tolerance - if dy is negative enough
    # Positive dy means hand pointing down (invalid)
    
    tolerance = config["angle_tolerance"]
    
    # We want hand pointing upward (dy should be negative)
    # For upward: dy < 0
    # For slightly tilted: still want dy < 0 with some tolerance
    if dy < 0:
        # Hand is pointing upward
        return True
    else:
        # Check if within tolerance (slightly tilted down but still acceptable)
        # The more negative dy is, the more upward
        # We accept if dy is not too positive
        return dy < tolerance

def count_extended_fingers_improved(landmarks):
    """
    IMPROVED finger counting - accurate palm detection
    Based on test.html logic which works perfectly
    """
    if not landmarks:
        return 0
    
    count = 0
    
    # Thumb detection based on hand orientation
    # Check if thumb tip is extended (horizontal position)
    is_right_hand = landmarks[5].x < landmarks[17].x
    
    if is_right_hand:
        # For right hand, thumb extended if tip is left of base
        if landmarks[4].x < landmarks[3].x - 0.02:
            count += 1
    else:
        # For left hand, thumb extended if tip is right of base
        if landmarks[4].x > landmarks[3].x + 0.02:
            count += 1
    
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
    
    return count

def is_palm_fully_open(landmarks):
    """
    Improved palm detection - all fingers extended
    Uses the same logic as test.html
    """
    if not landmarks:
        return False
    
    # Check all fingers are extended
    fingers_extended = 0
    
    # Thumb
    is_right_hand = landmarks[5].x < landmarks[17].x
    if is_right_hand:
        if landmarks[4].x < landmarks[3].x - 0.02:
            fingers_extended += 1
    else:
        if landmarks[4].x > landmarks[3].x + 0.02:
            fingers_extended += 1
    
    # Index finger
    if landmarks[8].y < landmarks[6].y - 0.03:
        fingers_extended += 1
    
    # Middle finger
    if landmarks[12].y < landmarks[10].y - 0.03:
        fingers_extended += 1
    
    # Ring finger
    if landmarks[16].y < landmarks[14].y - 0.02:
        fingers_extended += 1
    
    # Pinky
    if landmarks[20].y < landmarks[18].y - 0.02:
        fingers_extended += 1
    
    # Palm requires all 5 fingers extended
    return fingers_extended == 5

def get_gesture_info(finger_count, is_palm=False):
    """Return gesture name and icon for finger count"""
    if is_palm or finger_count == 5:
        return {"name": "Palm", "icon": "✋", "action": "toggle"}
    
    mapping = {
        0: {"name": "Fist", "icon": "✊", "action": "reset"},
        1: {"name": "1 Finger", "icon": "☝️", "action": "toggle"},
        2: {"name": "2 Fingers", "icon": "✌️", "action": "toggle"},
        3: {"name": "3 Fingers", "icon": "👌", "action": "toggle"},
        4: {"name": "4 Fingers", "icon": "🖖", "action": "toggle"},
    }
    
    if finger_count in mapping:
        return mapping[finger_count]
    return {"name": f"{finger_count} Fingers", "icon": "🖐️", "action": "toggle"}

# ========== HOME ASSISTANT WEBSOCKET ==========
class HAClient:
    """Home Assistant WebSocket Client"""
    
    def __init__(self, url, token, entities, cooldown_seconds, reset_gesture):
        self.url = url
        self.token = token
        self.entities = entities
        self.cooldown_seconds = cooldown_seconds
        self.reset_gesture = reset_gesture
        
        self.ws = None
        self.session = None
        self.connected = False
        self.last_gesture = None
        self.last_trigger_time = 0
        self.last_action = ""
        self.loop = None
        self.pending_gesture = None
        self.gesture_lock = threading.Lock()
        
        # Track toggled state for each entity
        self.entity_states = {i: False for i in range(1, 6)}
        
    def set_loop(self, loop):
        self.loop = loop
        
    async def connect(self):
        """Connect to Home Assistant WebSocket"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            logger.info(f"🔌 Connecting to HA at {self.url}")
            
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(self.url)
            
            msg = await self.ws.receive_json()
            logger.info(f"📨 HA initial message: {msg.get('type')}")
            
            if msg.get("type") != "auth_required":
                logger.error(f"❌ Expected auth_required, got: {msg}")
                self.connected = False
                return False
            
            auth_msg = {
                "type": "auth",
                "access_token": self.token
            }
            logger.info("🔐 Sending authentication...")
            await self.ws.send_json(auth_msg)
            
            msg = await self.ws.receive_json()
            logger.info(f"📨 HA auth response: {msg.get('type')}")
            
            if msg.get("type") == "auth_ok":
                self.connected = True
                logger.info("✅ HA WebSocket connected and authenticated")
                return True
            elif msg.get("type") == "auth_invalid":
                logger.error(f"❌ HA auth invalid: {msg.get('message', 'Invalid token')}")
                self.connected = False
                return False
            else:
                logger.error(f"❌ HA auth failed: {msg}")
                self.connected = False
                return False
                
        except Exception as e:
            logger.error(f"❌ HA connection error: {e}")
            self.connected = False
            return False
    
    async def toggle_entity(self, entity_num, entity_id):
        """Toggle an entity (on/off)"""
        if not self.connected or not self.ws:
            logger.warning("⚠️ Not connected to HA, cannot toggle entity")
            return False
        
        try:
            msg_id = int(time.time() * 1000)
            
            toggle_msg = {
                "id": msg_id,
                "type": "call_service",
                "domain": entity_id.split('.')[0],
                "service": "toggle",
                "service_data": {"entity_id": entity_id}
            }
            await self.ws.send_json(toggle_msg)
            
            # Update local state
            self.entity_states[entity_num] = not self.entity_states.get(entity_num, False)
            new_state = "ON" if self.entity_states[entity_num] else "OFF"
            
            logger.info(f"⚡ Toggled {entity_id} → {new_state}")
            self.last_action = f"Toggled {entity_id.split('.')[-1]} → {new_state}"
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to toggle entity: {e}")
            self.connected = False
            return False
    
    async def reset_gesture_state(self):
        """Reset all entities to OFF"""
        if not self.connected or not self.ws:
            logger.warning("⚠️ Not connected to HA, cannot reset")
            return False
        
        try:
            for i in range(1, 6):
                entity_id = self.entities.get(f"entity_{i}", "")
                if entity_id:
                    msg_id = int(time.time() * 1000)
                    turn_off_msg = {
                        "id": msg_id,
                        "type": "call_service",
                        "domain": entity_id.split('.')[0],
                        "service": "turn_off",
                        "service_data": {"entity_id": entity_id}
                    }
                    await self.ws.send_json(turn_off_msg)
                    self.entity_states[i] = False
            
            logger.info("🔄 Reset all entities to OFF")
            self.last_action = "Reset all entities"
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reset: {e}")
            self.connected = False
            return False
    
    async def handle_gesture(self, finger_count, is_palm=False):
        """Handle gesture and trigger HA actions with cooldown"""
        if not self.connected:
            return False
        
        # For palm, use 5 fingers
        if is_palm:
            finger_count = 5
        
        now = time.time()
        
        # Cooldown check
        with self.gesture_lock:
            if (self.last_gesture == finger_count and 
                now - self.last_trigger_time < self.cooldown_seconds):
                return False
        
        # Handle reset gesture (fist = 0 fingers)
        if finger_count == self.reset_gesture:
            logger.info(f"🎮 Reset gesture detected! Resetting all entities")
            with self.gesture_lock:
                self.last_gesture = finger_count
                self.last_trigger_time = now
            await self.reset_gesture_state()
            return True
        
        # Handle toggle gestures (1-5 fingers)
        if 1 <= finger_count <= 5:
            entity_key = f"entity_{finger_count}"
            entity_id = self.entities.get(entity_key, "")
            
            if entity_id:
                logger.info(f"🎮 {finger_count} finger(s) detected! Toggling {entity_id}")
                with self.gesture_lock:
                    self.last_gesture = finger_count
                    self.last_trigger_time = now
                await self.toggle_entity(finger_count, entity_id)
                return True
        
        return False
    
    async def disconnect(self):
        """Disconnect from Home Assistant"""
        if self.ws:
            await self.ws.close()
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False

# ========== RTSP PROCESSING THREAD ==========
class GestureProcessor:
    """Handles RTSP stream reading and gesture detection"""
    
    def __init__(self, ha_client, sio_server):
        self.ha_client = ha_client
        self.sio_server = sio_server
        self.running = False
        self.thread = None
        self.cap = None
        
        # Gesture stability - IMPROVED
        self.gesture_history = deque(maxlen=config["stability_frames"])
        self.current_stable_gesture = None
        self.current_finger_count = 0
        self.current_is_palm = False
        self.hand_detected = False
        self.last_stable_time = 0
        self.stability_counter = 0
        self.last_hand_removed_time = 0
        self.HAND_REMOVED_DELAY = 0.5  # 500ms delay before resetting after hand removal
        
        # Debug counters
        self.frame_count = 0
        self.detection_count = 0
        self.last_debug_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Async event loop reference
        self.loop = None
        
    def set_loop(self, loop):
        self.loop = loop
        
    def start(self):
        """Start the processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("📹 Camera processing thread started")
    
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("🛑 Gesture processor stopped")
    
    def _emit_async(self, event, data):
        """Emit Socket.IO event safely from thread"""
        if self.sio_server and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.sio_server.emit(event, data),
                self.loop
            )
    
    def _process_loop(self):
        """Main processing loop in separate thread"""
        global hands
        if hands is None:
            init_mediapipe()
        
        # Open RTSP stream with retry logic
        max_retries = 5
        retry_count = 0
        connected = False
        
        while self.running and retry_count < max_retries:
            logger.info(f"🔌 Connecting to RTSP stream: {config['rtsp_url']}")
            
            self.cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                retry_count += 1
                logger.error(f"❌ Failed to open RTSP stream (attempt {retry_count}/{max_retries})")
                self._emit_async("status", {
                    "status": "error",
                    "message": f"Failed to open RTSP stream (attempt {retry_count})"
                })
                time.sleep(5)
                continue
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, config["fps"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
            
            logger.info("✅ RTSP stream connected successfully")
            self._emit_async("status", {
                "status": "connected",
                "message": "Camera connected"
            })
            connected = True
            break
        
        if not connected:
            logger.error("❌ Could not connect to RTSP stream after retries")
            self._emit_async("status", {
                "status": "error",
                "message": "Could not connect to RTSP stream"
            })
            return
        
        # Process every frame for better detection
        process_every_n = max(1, 30 // config["fps"])
        frame_skip_counter = 0
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Read frame with timeout
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame, reconnecting...")
                    self._emit_async("status", {
                        "status": "warning",
                        "message": "Frame read failed, reconnecting..."
                    })
                    # Attempt to reconnect
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
                    continue
                
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                    
                    # Emit FPS for debugging
                    self._emit_async("fps_update", {"fps": self.current_fps})
                
                self.frame_count += 1
                
                # Debug every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.last_debug_time
                    logger.info(f"📊 Stats: FPS: {self.current_fps}, Detections: {self.detection_count}, Hand: {self.hand_detected}, Gesture: {self.current_stable_gesture}, HA: {'✓' if self.ha_client.connected else '✗'}")
                    self.last_debug_time = time.time()
                    self.detection_count = 0
                
                # Process frame
                frame_skip_counter += 1
                if frame_skip_counter >= process_every_n:
                    frame_skip_counter = 0
                    
                    try:
                        # Convert to RGB and flip horizontally for mirror effect
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame = cv2.flip(rgb_frame, 1)
                        
                        # Process with MediaPipe
                        results = hands.process(rgb_frame)
                        
                        finger_count = -1  # -1 means no hand detected
                        is_palm = False
                        hand_detected = False
                        hand_valid = False
                        
                        if results.multi_hand_landmarks:
                            hand_detected = True
                            self.detection_count += 1
                            
                            # Get first hand
                            landmarks = results.multi_hand_landmarks[0].landmark
                            
                            # IMPROVED: Check if hand is facing upward
                            is_upward = calculate_hand_angle(landmarks)
                            
                            if is_upward:
                                hand_valid = True
                                # Use improved finger counting
                                finger_count = count_extended_fingers_improved(landmarks)
                                is_palm = is_palm_fully_open(landmarks)
                                
                                # Log detection occasionally
                                if self.detection_count % 30 == 0:
                                    palm_str = " (PALM)" if is_palm else ""
                                    logger.info(f"✋ Hand detected - {finger_count} fingers{palm_str} (upward: ✓)")
                            else:
                                # Hand not facing upward - ignore
                                if self.detection_count % 50 == 0:
                                    logger.info(f"⚠️ Hand detected but not facing upward - ignoring")
                        
                        # IMPROVED: Handle hand removal without false triggers
                        now_time = time.time()
                        
                        if hand_detected and hand_valid:
                            # Hand is present and valid - reset removal timer
                            self.last_hand_removed_time = now_time
                            
                            # Update gesture history with valid finger count
                            self.gesture_history.append(finger_count if hand_detected else -1)
                            
                            # Find stable gesture
                            if len(self.gesture_history) == self.gesture_history.maxlen:
                                # Count valid gestures (ignore -1)
                                valid_gestures = [g for g in self.gesture_history if g >= 0]
                                if valid_gestures:
                                    # Get most common valid gesture
                                    gesture_counts = Counter(valid_gestures)
                                    most_common = gesture_counts.most_common(1)[0]
                                    stable_fingers = most_common[0]
                                    confidence = most_common[1] / len(self.gesture_history)
                                    
                                    # Check if this is a palm (all 5 fingers)
                                    stable_is_palm = is_palm and stable_fingers == 5
                                    
                                    # Only consider stable if confidence is high enough
                                    if confidence >= 0.6:
                                        # Check if gesture changed
                                        gesture_changed = (stable_fingers != self.current_stable_gesture or 
                                                          stable_is_palm != self.current_is_palm)
                                        
                                        if gesture_changed:
                                            self.current_stable_gesture = stable_fingers
                                            self.current_is_palm = stable_is_palm
                                            self.current_finger_count = stable_fingers
                                            self.hand_detected = True
                                            
                                            logger.info(f"🎭 Stable gesture detected: {stable_fingers} fingers{' (PALM)' if stable_is_palm else ''} (confidence: {confidence:.2f})")
                                            
                                            # Trigger HA action
                                            asyncio.run_coroutine_threadsafe(
                                                self.ha_client.handle_gesture(stable_fingers, stable_is_palm),
                                                self.ha_client.loop
                                            )
                                            
                                            # Emit gesture event
                                            gesture_info = get_gesture_info(stable_fingers, stable_is_palm)
                                            self._emit_async("gesture", {
                                                "fingerCount": stable_fingers,
                                                "gestureName": gesture_info["name"],
                                                "gestureIcon": gesture_info["icon"],
                                                "handDetected": True,
                                                "lastAction": self.ha_client.last_action,
                                                "haConnected": self.ha_client.connected,
                                                "confidence": confidence,
                                                "timestamp": datetime.now().isoformat()
                                            })
                                        else:
                                            self.current_finger_count = stable_fingers
                                            self.current_is_palm = stable_is_palm
                                            self.hand_detected = True
                                    else:
                                        # Use previous stable gesture
                                        self.current_finger_count = self.current_stable_gesture if self.current_stable_gesture is not None else 0
                                        self.current_is_palm = self.current_is_palm if self.current_stable_gesture is not None else False
                                        self.hand_detected = True
                            else:
                                # Not enough frames yet
                                self.current_finger_count = finger_count if hand_detected else 0
                                self.current_is_palm = is_palm
                                self.hand_detected = True
                        else:
                            # No hand detected or hand not valid
                            # Only reset after a delay to prevent false triggers during hand removal
                            if now_time - self.last_hand_removed_time > self.HAND_REMOVED_DELAY:
                                if self.current_stable_gesture is not None or self.hand_detected:
                                    logger.info("👋 Hand removed from frame - resetting gesture state")
                                    self.current_stable_gesture = None
                                    self.current_finger_count = 0
                                    self.current_is_palm = False
                                    self.hand_detected = False
                                    self.gesture_history.clear()
                                    
                                    # Send no-hand state
                                    self._emit_async("gesture_update", {
                                        "fingerCount": 0,
                                        "gestureName": "No Hand",
                                        "gestureIcon": "🤚",
                                        "handDetected": False,
                                        "lastAction": self.ha_client.last_action,
                                        "haConnected": self.ha_client.connected,
                                        "timestamp": datetime.now().isoformat(),
                                        "entityStates": self.ha_client.entity_states if self.ha_client else {}
                                    })
                            else:
                                # Keep current state during hand removal delay
                                pass
                        
                        # Broadcast current state for UI
                        if self.hand_detected and self.current_finger_count >= 0:
                            gesture_info = get_gesture_info(self.current_finger_count, self.current_is_palm)
                            gesture_name = gesture_info["name"]
                            gesture_icon = gesture_info["icon"]
                        else:
                            gesture_name = "No Hand"
                            gesture_icon = "🤚"
                        
                        self._emit_async("gesture_update", {
                            "fingerCount": self.current_finger_count if self.hand_detected else 0,
                            "gestureName": gesture_name,
                            "gestureIcon": gesture_icon,
                            "handDetected": self.hand_detected,
                            "lastAction": self.ha_client.last_action,
                            "haConnected": self.ha_client.connected,
                            "timestamp": datetime.now().isoformat(),
                            "entityStates": self.ha_client.entity_states if self.ha_client else {}
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Gesture detection error: {e}")
                        # Don't stop the loop on error
                        continue
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
                time.sleep(0.05)
        
        if self.cap:
            self.cap.release()
        logger.info("Camera loop ended")

# ========== HTTP SERVER ==========
class GestureServer:
    """HTTP Server with video streaming and WebSocket"""
    
    def __init__(self):
        self.app = aiohttp.web.Application()
        self.sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="aiohttp")
        self.sio.attach(self.app)
        
        self.ha_client = None
        self.gesture_processor = None
        self.ha_keepalive_task = None
        
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get("/", self.serve_index)
        self.app.router.add_get("/video", self.video_stream)
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/config", self.get_config)
        self.app.router.add_static("/static", "./web")
    
    def _setup_socket_events(self):
        """Setup Socket.IO event handlers"""
        @self.sio.on("connect")
        async def handle_connect(sid, environ):
            logger.info(f"🟢 Frontend connected: {sid}")
            await self.sio.emit("status", {
                "status": "connected",
                "haConnected": self.ha_client.connected if self.ha_client else False,
                "message": "Connected to server"
            })
            if self.ha_client:
                await self.sio.emit("entity_states", {
                    "states": self.ha_client.entity_states
                })
        
        @self.sio.on("disconnect")
        async def handle_disconnect(sid):
            logger.info(f"🔴 Frontend disconnected: {sid}")
        
        @self.sio.on("get_entities")
        async def handle_get_entities(sid):
            if self.ha_client:
                await self.sio.emit("entity_config", {
                    "entities": self.ha_client.entities
                })
    
    async def serve_index(self, request):
        """Serve the main HTML page"""
        index_path = Path(__file__).parent / "web" / "index.html"
        return aiohttp.web.FileResponse(index_path)
    
    async def video_stream(self, request):
        """Stream video from RTSP source with optimized settings"""
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-max_delay", "500000",
            "-i", config["rtsp_url"],
            "-c:v", "copy",
            "-f", "mp4",
            "-movflags", "frag_keyframe+empty_moov",
            "-tune", "zerolatency",
            "-an",
            "-"
        ]
        
        response = aiohttp.web.StreamResponse()
        response.headers["Content-Type"] = "video/mp4"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Connection"] = "keep-alive"
        await response.prepare(request)
        
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            while True:
                chunk = await process.stdout.read(8192)
                if not chunk:
                    break
                try:
                    await response.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    logger.info("Client disconnected from video stream")
                    break
            
            await process.wait()
            
        except asyncio.CancelledError:
            if process:
                process.terminate()
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            if process and process.returncode is None:
                process.terminate()
                try:
                    await process.wait()
                except:
                    pass
        
        return response
    
    async def health_check(self, request):
        """Health check endpoint"""
        return aiohttp.web.json_response({
            "status": "ok",
            "ha_connected": self.ha_client.connected if self.ha_client else False,
            "ha_url": self.ha_client.url if self.ha_client else None,
            "gesture_running": self.gesture_processor.running if self.gesture_processor else False,
            "current_gesture": self.gesture_processor.current_finger_count if self.gesture_processor else 0,
            "hand_detected": self.gesture_processor.hand_detected if self.gesture_processor else False,
            "fps": config["fps"],
            "current_fps": self.gesture_processor.current_fps if self.gesture_processor else 0,
            "uptime": datetime.now().isoformat()
        })
    
    async def get_config(self, request):
        """Get configuration endpoint"""
        return aiohttp.web.json_response({
            "fps": config["fps"],
            "frame_width": config["frame_width"],
            "frame_height": config["frame_height"],
            "cooldown_seconds": config["cooldown_seconds"],
            "detection_confidence": config["detection_confidence"],
            "tracking_confidence": config["tracking_confidence"],
            "reset_gesture": config["reset_gesture"],
            "stability_frames": config["stability_frames"],
            "angle_tolerance": config["angle_tolerance"],
            "entities": {
                "entity_1": config.get("entity_1", ""),
                "entity_2": config.get("entity_2", ""),
                "entity_3": config.get("entity_3", ""),
                "entity_4": config.get("entity_4", ""),
                "entity_5": config.get("entity_5", "")
            }
        })
    
    async def start(self):
        """Start the server and initialize components"""
        # Print startup banner
        logger.info("=" * 60)
        logger.info("🎮 Starting Gesture Control Add-on v3.0")
        logger.info("=" * 60)
        logger.info(f"📹 RTSP URL: {'✅ Configured' if config['rtsp_url'] != 'rtsp://your-camera-ip:554/stream' else '⚠️ Using default'}")
        logger.info(f"💡 Entity 1: {config.get('entity_1', 'Not configured')}")
        logger.info(f"💡 Entity 2: {config.get('entity_2', 'Not configured')}")
        logger.info(f"💡 Entity 3: {config.get('entity_3', 'Not configured')}")
        logger.info(f"💡 Entity 4: {config.get('entity_4', 'Not configured')}")
        logger.info(f"💡 Entity 5: {config.get('entity_5', 'Not configured')}")
        logger.info(f"🔄 Reset Gesture: {config['reset_gesture']} fingers")
        logger.info(f"⏱️  Cooldown: {config['cooldown_seconds']}s")
        logger.info(f"🎯 Detection Confidence: {config['detection_confidence']}")
        logger.info(f"🎯 Tracking Confidence: {config['tracking_confidence']}")
        logger.info(f"📐 Resolution: {config['frame_width']}x{config['frame_height']}")
        logger.info(f"⚡ Target FPS: {config['fps']}")
        logger.info(f"📐 Angle Tolerance: {config['angle_tolerance']}")
        logger.info(f"🔌 HA URL: {config['ha_url']}")
        logger.info("=" * 60)
        
        # Initialize HA client
        entities = {
            "entity_1": config.get("entity_1", ""),
            "entity_2": config.get("entity_2", ""),
            "entity_3": config.get("entity_3", ""),
            "entity_4": config.get("entity_4", ""),
            "entity_5": config.get("entity_5", "")
        }
        
        self.ha_client = HAClient(
            config["ha_url"],
            config["ha_token"],
            entities,
            config["cooldown_seconds"],
            config["reset_gesture"]
        )
        self.ha_client.set_loop(asyncio.get_event_loop())
        
        # Connect to HA
        connected = await self.ha_client.connect()
        if not connected:
            logger.error("❌ Failed to connect to HA! Check your token and URL")
        
        # Start gesture processor in background thread
        self.gesture_processor = GestureProcessor(self.ha_client, self.sio)
        self.gesture_processor.set_loop(asyncio.get_event_loop())
        self.gesture_processor.start()
        
        # Start background HA keepalive
        self.ha_keepalive_task = asyncio.create_task(self._ha_keepalive())
        
        # Run HTTP server
        runner = aiohttp.web.AppRunner(self.app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "0.0.0.0", 8099)
        await site.start()
        
        logger.info("=" * 60)
        logger.info("🌐 Web server started on port 8099")
        logger.info("📍 Access the UI at: http://[YOUR-IP]:8099/")
        logger.info("🔍 Health check: http://[YOUR-IP]:8099/health")
        logger.info("=" * 60)
        
        # Keep running
        await asyncio.Event().wait()
    
    async def _ha_keepalive(self):
        """Keep HA connection alive and reconnect if needed"""
        while True:
            await asyncio.sleep(30)
            if self.ha_client and not self.ha_client.connected:
                logger.info("🔄 Reconnecting to HA...")
                await self.ha_client.connect()
                if self.gesture_processor:
                    await self.sio.emit("status", {
                        "status": "ha_reconnected",
                        "haConnected": self.ha_client.connected,
                        "message": "Reconnected to Home Assistant"
                    })

# ========== MAIN ENTRY POINT ==========
def main():
    """Main entry point"""
    server = GestureServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        if hasattr(server, 'gesture_processor') and server.gesture_processor:
            server.gesture_processor.stop()
        if hasattr(server, 'ha_client') and server.ha_client:
            asyncio.run(server.ha_client.disconnect())
        logger.info("👋 Goodbye!")

if __name__ == "__main__":
    main()