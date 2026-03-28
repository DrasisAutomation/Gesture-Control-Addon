#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
Enhanced with multi-person filtering, improved palm detection, orientation filtering,
and stable gesture logic.
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from collections import Counter
import math

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
        # Development fallback
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
            "reset_gesture": 0,  # Not used now, fist does nothing
            "stability_frames": 5,
            "center_region_size": 0.3,  # 30% of frame from center
            "hand_orientation_threshold": 30,  # degrees from vertical
            "movement_threshold": 0.05,  # hand movement threshold
            "stationary_frames": 10,  # frames hand must be stationary
        }
    
    # Ensure all keys exist
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
        "center_region_size": 0.3,
        "hand_orientation_threshold": 30,
        "movement_threshold": 0.05,
        "stationary_frames": 10,
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

# ========== GESTURE DETECTION ==========
mp_hands = mp.solutions.hands
hands = None

def init_mediapipe():
    """Initialize MediaPipe Hands"""
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Allow up to 2 hands for selection
        min_detection_confidence=config["detection_confidence"],
        min_tracking_confidence=config["tracking_confidence"],
        model_complexity=0
    )
    logger.info("✅ MediaPipe initialized")

def calculate_hand_orientation(landmarks):
    """
    Calculate hand orientation angle from vertical.
    Returns angle in degrees (0 = vertical upward, 90 = horizontal)
    """
    # Use wrist (0) and middle finger MCP (9) to get hand direction vector
    wrist = landmarks[0]
    middle_base = landmarks[9]
    
    # Vector from wrist to middle finger base
    dx = middle_base.x - wrist.x
    dy = middle_base.y - wrist.y
    
    # Calculate angle from vertical (pointing up)
    # Vertical vector is (0, -1) in image coordinates (y increases downward)
    # So we need to handle coordinate system: y increases downward
    # For upward orientation, wrist should be below fingers (dy negative)
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    
    # Check if hand is facing upward (fingers above wrist)
    # In image coordinates, fingers have lower y than wrist
    finger_y_avg = sum(landmarks[i].y for i in [8, 12, 16, 20]) / 4
    wrist_y = wrist.y
    
    if finger_y_avg > wrist_y:
        # Hand is pointing downward, not valid
        return 180
    
    return angle

def is_hand_in_center(landmarks, frame_width, frame_height, center_region_size):
    """Check if hand's center is within the center region of the frame"""
    # Calculate hand center (average of all landmarks)
    hand_center_x = sum(lm.x for lm in landmarks) / len(landmarks)
    hand_center_y = sum(lm.y for lm in landmarks) / len(landmarks)
    
    # Define center region bounds
    region_min_x = 0.5 - (center_region_size / 2)
    region_max_x = 0.5 + (center_region_size / 2)
    region_min_y = 0.5 - (center_region_size / 2)
    region_max_y = 0.5 + (center_region_size / 2)
    
    return (region_min_x <= hand_center_x <= region_max_x and
            region_min_y <= hand_center_y <= region_max_y)

def calculate_hand_movement(prev_landmarks, curr_landmarks):
    """Calculate average movement distance between two hand positions"""
    if prev_landmarks is None or curr_landmarks is None:
        return 1.0  # Large movement if no previous data
    
    total_distance = 0
    for i in range(min(len(prev_landmarks), len(curr_landmarks))):
        dx = prev_landmarks[i].x - curr_landmarks[i].x
        dy = prev_landmarks[i].y - curr_landmarks[i].y
        total_distance += math.sqrt(dx*dx + dy*dy)
    
    return total_distance / min(len(prev_landmarks), len(curr_landmarks))

def count_extended_fingers(landmarks, image_width, image_height):
    """
    Count number of extended fingers from hand landmarks.
    Improved detection for palm (all fingers extended) using better thresholds.
    Reference from test.html which works correctly.
    """
    if not landmarks:
        return -1
    
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
    
    # Thumb detection (horizontal position based on hand orientation)
    is_right_hand = landmarks[5].x < landmarks[17].x
    if is_right_hand:
        if landmarks[4].x < landmarks[3].x - 0.02:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x + 0.02:
            count += 1
    
    return min(count, 5)

def is_valid_palm(landmarks):
    """
    Enhanced palm detection that correctly identifies full palm (5 fingers extended)
    """
    # Count fingers using the improved method
    finger_count = 0
    
    # Check all four fingers (index, middle, ring, pinky)
    # More generous threshold for palm detection
    if landmarks[8].y < landmarks[6].y - 0.02:
        finger_count += 1
    if landmarks[12].y < landmarks[10].y - 0.02:
        finger_count += 1
    if landmarks[16].y < landmarks[14].y - 0.015:
        finger_count += 1
    if landmarks[20].y < landmarks[18].y - 0.015:
        finger_count += 1
    
    # Thumb detection - more generous for palm
    is_right_hand = landmarks[5].x < landmarks[17].x
    thumb_extended = False
    if is_right_hand:
        if landmarks[4].x < landmarks[3].x - 0.015:
            thumb_extended = True
    else:
        if landmarks[4].x > landmarks[3].x + 0.015:
            thumb_extended = True
    
    if thumb_extended:
        finger_count += 1
    
    # Check spread between fingers (palm should have fingers spread apart)
    # Calculate spread between index and pinky
    index_tip = landmarks[8]
    pinky_tip = landmarks[20]
    finger_spread = abs(index_tip.x - pinky_tip.x)
    
    # Palm is true if all 5 fingers are extended AND fingers are spread
    if finger_count == 5 and finger_spread > 0.15:
        return True, 5
    
    return False, finger_count

def get_gesture_info(finger_count):
    """Return gesture name and icon for finger count"""
    mapping = {
        1: {"name": "1 Finger", "icon": "☝️", "action": "toggle", "entity": 1},
        2: {"name": "2 Fingers", "icon": "✌️", "action": "toggle", "entity": 2},
        3: {"name": "3 Fingers", "icon": "👌", "action": "toggle", "entity": 3},
        4: {"name": "4 Fingers", "icon": "🖖", "action": "toggle", "entity": 4},
        5: {"name": "Palm", "icon": "✋", "action": "toggle", "entity": 5}
    }
    if finger_count in mapping:
        return mapping[finger_count]
    return {"name": f"{finger_count} Fingers", "icon": "🖐️", "action": "none", "entity": None}

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
        self.reconnect_task = None
        
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
                
                for i in range(1, 6):
                    entity_id = self.entities.get(f"entity_{i}", "")
                    if entity_id:
                        await self.subscribe_to_entity(entity_id)
                
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
    
    async def subscribe_to_entity(self, entity_id):
        """Subscribe to entity state changes"""
        if not self.connected or not self.ws:
            return
        
        try:
            msg_id = int(time.time() * 1000)
            subscribe_msg = {
                "id": msg_id,
                "type": "subscribe_events",
                "event_type": "state_changed"
            }
            await self.ws.send_json(subscribe_msg)
            logger.info(f"📡 Subscribed to state changes for {entity_id}")
        except Exception as e:
            logger.error(f"❌ Failed to subscribe: {e}")
    
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
            
            self.entity_states[entity_num] = not self.entity_states.get(entity_num, False)
            new_state = "ON" if self.entity_states[entity_num] else "OFF"
            
            logger.info(f"⚡ Toggled {entity_id} → {new_state}")
            self.last_action = f"Toggled {entity_id.split('.')[-1]} → {new_state}"
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to toggle entity: {e}")
            self.connected = False
            return False
    
    async def handle_gesture(self, finger_count):
        """Handle gesture and trigger HA actions with cooldown"""
        if not self.connected:
            logger.debug(f"⚠️ Not connected to HA, ignoring gesture: {finger_count} fingers")
            return False
            
        # Fist (0 fingers) does nothing - no action
        if finger_count == 0:
            logger.debug("Fist detected - no action (fist disabled)")
            return False
            
        now = time.time()
        
        # Cooldown check (only if same gesture)
        if (self.last_gesture == finger_count and 
            now - self.last_trigger_time < self.cooldown_seconds):
            logger.debug(f"⏱️ Cooldown active for gesture: {finger_count}")
            return False
        
        # Handle toggle gestures (1-5 fingers)
        if 1 <= finger_count <= 5:
            entity_key = f"entity_{finger_count}"
            entity_id = self.entities.get(entity_key, "")
            
            if entity_id:
                logger.info(f"🎮 {finger_count} finger(s) detected! Toggling {entity_id}")
                self.last_gesture = finger_count
                self.last_trigger_time = now
                await self.toggle_entity(finger_count, entity_id)
                return True
            else:
                logger.warning(f"⚠️ No entity configured for {finger_count} finger(s)")
                return False
        
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
    """Handles RTSP stream reading and gesture detection with improved stability"""
    
    def __init__(self, ha_client, sio_server):
        self.ha_client = ha_client
        self.sio_server = sio_server
        self.running = False
        self.thread = None
        self.cap = None
        
        # Gesture stability
        self.gesture_history = deque(maxlen=config["stability_frames"])
        self.current_stable_gesture = None
        self.current_finger_count = 0
        self.hand_detected = False
        self.last_stable_time = 0
        self.stability_counter = 0
        
        # Hand tracking for movement filtering
        self.prev_landmarks = None
        self.stationary_counter = 0
        self.hand_position_history = deque(maxlen=config["stationary_frames"])
        
        # Debug counters
        self.frame_count = 0
        self.detection_count = 0
        self.last_debug_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.processing_time = 0
        
        # Async event loop reference
        self.loop = None
        
        # Track if gesture was already triggered
        self.last_triggered_gesture = None
        self.trigger_frame_counter = 0
        
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
    
    def _select_best_hand(self, multi_hand_landmarks, frame_shape):
        """
        Select the best hand to track based on:
        1. Hand in center region
        2. Hand orientation (prefer upward facing)
        Returns selected hand landmarks or None
        """
        if not multi_hand_landmarks:
            return None
        
        best_hand = None
        best_score = -1
        
        for hand_landmarks in multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Check if hand is in center region
            if not is_hand_in_center(landmarks, frame_shape[1], frame_shape[0], 
                                     config["center_region_size"]):
                continue
            
            # Check hand orientation (must be upward)
            orientation = calculate_hand_orientation(landmarks)
            if orientation > config["hand_orientation_threshold"]:
                continue
            
            # Calculate score: center proximity + orientation
            hand_center_x = sum(lm.x for lm in landmarks) / len(landmarks)
            hand_center_y = sum(lm.y for lm in landmarks) / len(landmarks)
            center_distance = abs(hand_center_x - 0.5) + abs(hand_center_y - 0.5)
            center_score = 1 - (center_distance / 0.5)
            orientation_score = 1 - (orientation / 90)
            
            total_score = center_score * 0.6 + orientation_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_hand = landmarks
        
        return best_hand
    
    def _is_hand_stationary(self, current_landmarks):
        """Check if hand movement is below threshold"""
        if current_landmarks is None:
            return False
        
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return False
        
        movement = calculate_hand_movement(self.prev_landmarks, current_landmarks)
        
        if movement < config["movement_threshold"]:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0
        
        self.prev_landmarks = current_landmarks
        return self.stationary_counter >= config["stationary_frames"]
    
    def _process_loop(self):
        """Main processing loop in separate thread"""
        global hands
        if hands is None:
            init_mediapipe()
        
        logger.info(f"🔌 Connecting to RTSP stream: {config['rtsp_url']}")
        
        self.cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logger.error("❌ Failed to open RTSP stream")
            self._emit_async("status", {
                "status": "error",
                "message": "Failed to open RTSP stream"
            })
            return
        
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
        
        process_every_n = max(1, 30 // config["fps"])
        frame_skip_counter = 0
        
        while self.running:
            try:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame")
                    time.sleep(0.05)
                    continue
                
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                self.frame_count += 1
                
                # Debug every 200 frames
                if self.frame_count % 200 == 0:
                    elapsed = time.time() - self.last_debug_time
                    logger.info(f"📊 Stats: FPS: {self.current_fps}, Processing: {self.processing_time*1000:.1f}ms, Detections: {self.detection_count}, Hand: {self.hand_detected}, Gesture: {self.current_stable_gesture}, HA: {'✓' if self.ha_client.connected else '✗'}")
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
                        
                        finger_count = 0
                        self.hand_detected = False
                        hand_landmarks = None
                        is_stationary = False
                        
                        if results.multi_hand_landmarks:
                            # Select best hand based on position and orientation
                            hand_landmarks = self._select_best_hand(results.multi_hand_landmarks, frame.shape)
                            
                            if hand_landmarks:
                                self.hand_detected = True
                                self.detection_count += 1
                                
                                # Check if hand is stationary
                                is_stationary = self._is_hand_stationary(hand_landmarks)
                                
                                # Check if it's a valid palm
                                is_palm, palm_fingers = is_valid_palm(hand_landmarks)
                                
                                if is_palm:
                                    finger_count = 5
                                else:
                                    finger_count = count_extended_fingers(hand_landmarks, 
                                                                          frame.shape[1], 
                                                                          frame.shape[0])
                                
                                # Log detection occasionally
                                if self.detection_count % 30 == 0:
                                    orientation = calculate_hand_orientation(hand_landmarks)
                                    logger.info(f"✋ Hand detected - {finger_count} fingers, orientation: {orientation:.1f}°, stationary: {is_stationary}")
                        
                        # Only process gestures if hand is stationary
                        if self.hand_detected and is_stationary:
                            # Update gesture history
                            self.gesture_history.append(finger_count)
                            
                            # Find stable gesture
                            if len(self.gesture_history) == self.gesture_history.maxlen:
                                # Count valid gestures
                                valid_gestures = [g for g in self.gesture_history if g >= 0]
                                if valid_gestures:
                                    # Get most common valid gesture
                                    gesture_counts = Counter(valid_gestures)
                                    most_common = gesture_counts.most_common(1)[0]
                                    stable = most_common[0]
                                    confidence = most_common[1] / len(self.gesture_history)
                                    
                                    # Only consider stable if confidence is high enough
                                    if confidence >= 0.6:
                                        if stable != self.current_stable_gesture:
                                            self.current_stable_gesture = stable
                                            self.current_finger_count = stable
                                            self.trigger_frame_counter = 0
                                            logger.info(f"🎭 Stable gesture detected: {stable} fingers (confidence: {confidence:.2f})")
                                            
                                            # Trigger HA action only once per stable gesture
                                            if stable != self.last_triggered_gesture:
                                                self.last_triggered_gesture = stable
                                                asyncio.run_coroutine_threadsafe(
                                                    self.ha_client.handle_gesture(stable),
                                                    self.ha_client.loop
                                                )
                                            else:
                                                # Increment trigger frame counter to prevent repeated triggers
                                                self.trigger_frame_counter += 1
                                                if self.trigger_frame_counter > 10:
                                                    self.last_triggered_gesture = None
                                                    self.trigger_frame_counter = 0
                                            
                                            # Emit gesture event
                                            gesture_info = get_gesture_info(stable)
                                            self._emit_async("gesture", {
                                                "fingerCount": stable,
                                                "gestureName": gesture_info["name"],
                                                "gestureIcon": gesture_info["icon"],
                                                "handDetected": True,
                                                "lastAction": self.ha_client.last_action,
                                                "haConnected": self.ha_client.connected,
                                                "confidence": confidence,
                                                "stationary": True,
                                                "timestamp": datetime.now().isoformat()
                                            })
                                        else:
                                            self.current_finger_count = stable
                                    else:
                                        self.current_finger_count = self.current_stable_gesture if self.current_stable_gesture is not None else 0
                                else:
                                    # No valid gesture
                                    if self.current_stable_gesture is not None:
                                        logger.info("👋 Hand removed or invalid")
                                        self.current_stable_gesture = None
                                        self.last_triggered_gesture = None
                                    self.current_finger_count = 0
                                    self.hand_detected = False
                        else:
                            # Hand not stationary or no hand detected
                            if self.hand_detected and not is_stationary:
                                # Hand moving - reset stable gesture
                                if self.current_stable_gesture is not None:
                                    self.current_stable_gesture = None
                                    self.last_triggered_gesture = None
                                    logger.debug("Hand moving - resetting gesture")
                            
                            # Clear history when hand is not stationary
                            if not self.hand_detected or not is_stationary:
                                self.gesture_history.clear()
                                self.current_stable_gesture = None
                                self.current_finger_count = 0
                                self.stationary_counter = 0
                        
                        # Broadcast current state for UI
                        if self.hand_detected and self.current_finger_count >= 0:
                            gesture_info = get_gesture_info(self.current_finger_count)
                            gesture_name = gesture_info["name"]
                            gesture_icon = gesture_info["icon"]
                        else:
                            gesture_name = "No Hand"
                            gesture_icon = "🤚"
                            self.current_finger_count = 0
                        
                        self._emit_async("gesture_update", {
                            "fingerCount": self.current_finger_count if self.hand_detected else 0,
                            "gestureName": gesture_name,
                            "gestureIcon": gesture_icon,
                            "handDetected": self.hand_detected,
                            "stationary": is_stationary,
                            "lastAction": self.ha_client.last_action,
                            "haConnected": self.ha_client.connected,
                            "timestamp": datetime.now().isoformat(),
                            "entityStates": self.ha_client.entity_states if self.ha_client else {}
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Gesture detection error: {e}")
                
                # Calculate processing time
                self.processing_time = time.time() - frame_start
                
                # Ensure we're not processing too fast to prevent CPU issues
                if self.processing_time < 0.01:
                    time.sleep(0.005)
                
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
                time.sleep(0.05)
        
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
            "hand_stationary": self.gesture_processor.stationary_counter >= config["stationary_frames"] if self.gesture_processor else False,
            "fps": config["fps"],
            "current_fps": self.gesture_processor.current_fps if self.gesture_processor else 0,
            "processing_time_ms": self.gesture_processor.processing_time * 1000 if self.gesture_processor else 0,
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
            "center_region_size": config["center_region_size"],
            "hand_orientation_threshold": config["hand_orientation_threshold"],
            "movement_threshold": config["movement_threshold"],
            "stationary_frames": config["stationary_frames"],
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
        logger.info("🎮 Starting Gesture Control Add-on v2.1 (Enhanced)")
        logger.info("=" * 60)
        logger.info(f"📹 RTSP URL: {'✅ Configured' if config['rtsp_url'] != 'rtsp://your-camera-ip:554/stream' else '⚠️ Using default'}")
        logger.info(f"💡 Entity 1: {config.get('entity_1', 'Not configured')}")
        logger.info(f"💡 Entity 2: {config.get('entity_2', 'Not configured')}")
        logger.info(f"💡 Entity 3: {config.get('entity_3', 'Not configured')}")
        logger.info(f"💡 Entity 4: {config.get('entity_4', 'Not configured')}")
        logger.info(f"💡 Entity 5: {config.get('entity_5', 'Not configured')}")
        logger.info(f"⏱️  Cooldown: {config['cooldown_seconds']}s")
        logger.info(f"🎯 Detection Confidence: {config['detection_confidence']}")
        logger.info(f"🎯 Tracking Confidence: {config['tracking_confidence']}")
        logger.info(f"📐 Resolution: {config['frame_width']}x{config['frame_height']}")
        logger.info(f"⚡ Target FPS: {config['fps']}")
        logger.info(f"🎯 Center Region: {config['center_region_size']*100}% from center")
        logger.info(f"🎯 Orientation Threshold: {config['hand_orientation_threshold']}° from vertical")
        logger.info(f"🎯 Movement Threshold: {config['movement_threshold']}")
        logger.info(f"🎯 Stationary Frames: {config['stationary_frames']}")
        logger.info(f"🔌 HA URL: {config['ha_url']}")
        logger.info(f"🔑 HA Token: {config['ha_token'][:20]}..." if config['ha_token'] else "⚠️ No token set")
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
        
        # Start gesture processor
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