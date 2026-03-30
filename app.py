#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
Enhanced with improved gesture detection, stability, and accuracy
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
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
    logger = logging.getLogger("gesture-control")
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
            "cooldown_seconds": 2.0,
            "detection_confidence": 0.7,
            "tracking_confidence": 0.6,
            "reset_gesture": 0,
            "stability_frames": 8,
            "gesture_hold_time": 0.5,
            "min_hand_size": 0.05,
            "max_hands_to_track": 2,
        }

    defaults = {
        "fps": 15,
        "frame_width": 640,
        "frame_height": 480,
        "bitrate": "500k",
        "cooldown_seconds": 2.0,
        "detection_confidence": 0.7,
        "tracking_confidence": 0.6,
        "reset_gesture": 0,
        "stability_frames": 8,
        "gesture_hold_time": 0.5,
        "min_hand_size": 0.05,
        "max_hands_to_track": 2,
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
    """Initialize MediaPipe Hands with optimized settings"""
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config["max_hands_to_track"],
        min_detection_confidence=config["detection_confidence"],
        min_tracking_confidence=config["tracking_confidence"],
        model_complexity=1  # Higher complexity for better accuracy
    )
    logger.info("✅ MediaPipe initialized with enhanced settings")


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in degrees"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_hand_size(landmarks):
    """Calculate hand size based on wrist to middle finger tip distance"""
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    return calculate_distance(wrist, middle_tip)


def get_hand_center(landmarks):
    """Get the center point of the hand (palm center approximation)"""
    # Use landmarks 0 (wrist), 5, 9, 13, 17 (finger bases) for palm center
    palm_landmarks = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    center_x = sum(l.x for l in palm_landmarks) / len(palm_landmarks)
    center_y = sum(l.y for l in palm_landmarks) / len(palm_landmarks)
    return center_x, center_y


def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx, 
                       wrist_idx=0, is_thumb=False, handedness="Right"):
    """
    Determine if a finger is extended using multiple criteria:
    - Tip position relative to PIP joint
    - Angle at PIP joint (straighter = more extended)
    - Distance ratios
    """
    tip = landmarks[finger_tip_idx]
    pip = landmarks[finger_pip_idx]
    mcp = landmarks[finger_mcp_idx]
    wrist = landmarks[wrist_idx]
    
    if is_thumb:
        # Thumb uses different logic based on handedness
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        
        # Calculate thumb extension based on horizontal spread
        if handedness == "Right":
            # Right hand: thumb extended if tip is significantly left of IP joint
            horizontal_extension = thumb_tip.x < thumb_ip.x - 0.03
        else:
            # Left hand: thumb extended if tip is significantly right of IP joint
            horizontal_extension = thumb_tip.x > thumb_ip.x + 0.03
        
        # Also check vertical extension and distance from palm
        thumb_palm_distance = calculate_distance(thumb_tip, index_mcp)
        thumb_extended_by_distance = thumb_palm_distance > 0.1
        
        # Thumb is extended if either condition is met
        return horizontal_extension or thumb_extended_by_distance
    
    else:
        # For other fingers, use multiple criteria
        
        # Criterion 1: Tip above PIP (in image coordinates, lower y = higher)
        tip_above_pip = tip.y < pip.y - 0.02
        
        # Criterion 2: Angle at PIP joint (extended finger has angle > 150 degrees)
        pip_angle = calculate_angle(mcp, pip, tip)
        angle_extended = pip_angle > 140
        
        # Criterion 3: Tip-to-wrist distance vs MCP-to-wrist distance
        tip_wrist_dist = calculate_distance(tip, wrist)
        mcp_wrist_dist = calculate_distance(mcp, wrist)
        distance_extended = tip_wrist_dist > mcp_wrist_dist * 1.1
        
        # Finger is extended if at least 2 of 3 criteria are met
        criteria_met = sum([tip_above_pip, angle_extended, distance_extended])
        return criteria_met >= 2


def count_extended_fingers(landmarks, handedness="Right"):
    """
    Count number of extended fingers with improved accuracy.
    Returns tuple: (finger_count, finger_states_dict)
    """
    if not landmarks:
        return 0, {}

    finger_states = {}
    
    # Thumb (index 4, 3, 2)
    thumb_extended = is_finger_extended(
        landmarks, 4, 3, 2, is_thumb=True, handedness=handedness
    )
    finger_states["thumb"] = thumb_extended
    
    # Index finger (tip=8, pip=6, mcp=5)
    index_extended = is_finger_extended(landmarks, 8, 6, 5)
    finger_states["index"] = index_extended
    
    # Middle finger (tip=12, pip=10, mcp=9)
    middle_extended = is_finger_extended(landmarks, 12, 10, 9)
    finger_states["middle"] = middle_extended
    
    # Ring finger (tip=16, pip=14, mcp=13)
    ring_extended = is_finger_extended(landmarks, 16, 14, 13)
    finger_states["ring"] = ring_extended
    
    # Pinky finger (tip=20, pip=18, mcp=17)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 17)
    finger_states["pinky"] = pinky_extended
    
    total = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    return total, finger_states


def detect_palm_gesture(landmarks, finger_count, finger_states):
    """
    Special detection for open palm (5 fingers) vs 4 fingers.
    Uses additional checks for palm openness.
    """
    if finger_count < 4:
        return finger_count
    
    # Check finger spread - open palm has spread fingers
    index_tip = landmarks[8]
    pinky_tip = landmarks[20]
    middle_tip = landmarks[12]
    wrist = landmarks[0]
    
    # Horizontal spread between index and pinky tips
    finger_spread = abs(index_tip.x - pinky_tip.x)
    
    # Hand height (wrist to middle finger tip)
    hand_height = abs(middle_tip.y - wrist.y)
    
    # Spread ratio - open palm typically has high spread relative to height
    spread_ratio = finger_spread / (hand_height + 1e-6)
    
    # Check if all 5 fingers appear extended (relaxed criteria for palm)
    all_fingers_up = all([
        finger_states.get("thumb", False),
        finger_states.get("index", False),
        finger_states.get("middle", False),
        finger_states.get("ring", False),
        finger_states.get("pinky", False)
    ])
    
    # Check thumb-index gap (open palm has larger gap)
    thumb_tip = landmarks[4]
    thumb_index_gap = calculate_distance(thumb_tip, index_tip)
    
    # Decision logic for 5 vs 4 fingers
    if finger_count == 4:
        # Check if it might actually be 5 fingers
        if spread_ratio > 0.6 and thumb_index_gap > 0.15:
            # Likely an open palm misdetected as 4
            logger.debug(f"Correcting 4→5 fingers (spread={spread_ratio:.2f}, gap={thumb_index_gap:.2f})")
            return 5
    
    if finger_count == 5:
        # Verify it's really an open palm
        if spread_ratio < 0.3:
            # Fingers too close together, might be 4
            logger.debug(f"Correcting 5→4 fingers (low spread={spread_ratio:.2f})")
            return 4
    
    return finger_count


def get_gesture_info(finger_count):
    """Return gesture name and icon for finger count"""
    mapping = {
        0: {"name": "Fist (Reset)", "icon": "✊", "action": "reset"},
        1: {"name": "1 Finger", "icon": "☝️", "action": "toggle"},
        2: {"name": "2 Fingers", "icon": "✌️", "action": "toggle"},
        3: {"name": "3 Fingers", "icon": "👌", "action": "toggle"},
        4: {"name": "4 Fingers", "icon": "🖖", "action": "toggle"},
        5: {"name": "Open Palm", "icon": "✋", "action": "toggle"}
    }
    return mapping.get(finger_count, {"name": f"{finger_count} Fingers", "icon": "🖐️", "action": "toggle"})


# ========== HOME ASSISTANT WEBSOCKET ==========
class HAClient:
    """Home Assistant WebSocket Client with robust connection handling"""

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
        self.message_id = 1

        # Track toggled state for each entity
        self.entity_states = {i: False for i in range(1, 6)}
        
        # Gesture locking - prevents repeated triggers
        self.gesture_locked = False
        self.locked_gesture = None

    def set_loop(self, loop):
        self.loop = loop

    def get_next_id(self):
        """Get next message ID for HA websocket"""
        self.message_id += 1
        return self.message_id

    async def connect(self):
        """Connect to Home Assistant WebSocket with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.session and not self.session.closed:
                    await self.session.close()

                logger.info(f"🔌 Connecting to HA at {self.url} (attempt {attempt + 1}/{max_retries})")

                timeout = aiohttp.ClientTimeout(total=30)
                self.session = aiohttp.ClientSession(timeout=timeout)
                self.ws = await self.session.ws_connect(self.url, heartbeat=30)

                msg = await asyncio.wait_for(self.ws.receive_json(), timeout=10)
                logger.info(f"📨 HA initial message: {msg.get('type')}")

                if msg.get("type") != "auth_required":
                    logger.error(f"❌ Expected auth_required, got: {msg}")
                    continue

                auth_msg = {"type": "auth", "access_token": self.token}
                await self.ws.send_json(auth_msg)

                msg = await asyncio.wait_for(self.ws.receive_json(), timeout=10)
                logger.info(f"📨 HA auth response: {msg.get('type')}")

                if msg.get("type") == "auth_ok":
                    self.connected = True
                    logger.info("✅ HA WebSocket connected and authenticated")
                    return True
                elif msg.get("type") == "auth_invalid":
                    logger.error(f"❌ HA auth invalid: {msg.get('message', 'Invalid token')}")
                    self.connected = False
                    return False

            except asyncio.TimeoutError:
                logger.error(f"❌ HA connection timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"❌ HA connection error: {e}")

            await asyncio.sleep(2 ** attempt)

        self.connected = False
        return False

    async def toggle_entity(self, entity_num, entity_id):
        """Toggle an entity (on/off)"""
        if not self.connected or not self.ws:
            logger.warning("⚠️ Not connected to HA, cannot toggle entity")
            return False

        try:
            domain = entity_id.split('.')[0]
            toggle_msg = {
                "id": self.get_next_id(),
                "type": "call_service",
                "domain": domain,
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

    async def reset_gesture_state(self):
        """Reset all entities to OFF"""
        if not self.connected or not self.ws:
            return False

        try:
            for i in range(1, 6):
                entity_id = self.entities.get(f"entity_{i}", "")
                if entity_id:
                    domain = entity_id.split('.')[0]
                    turn_off_msg = {
                        "id": self.get_next_id(),
                        "type": "call_service",
                        "domain": domain,
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

    def unlock_gesture(self):
        """Unlock gesture when hand is removed"""
        if self.gesture_locked:
            logger.info(f"🔓 Gesture unlocked (was: {self.locked_gesture})")
        self.gesture_locked = False
        self.locked_gesture = None

    async def handle_gesture(self, finger_count):
        """Handle gesture with locking to prevent repeated triggers"""
        if not self.connected:
            return False

        now = time.time()

        # Check if this gesture is locked (already triggered and hand still present)
        if self.gesture_locked and self.locked_gesture == finger_count:
            logger.debug(f"🔒 Gesture {finger_count} is locked, ignoring")
            return False

        # Cooldown check (time-based, for different gestures)
        if now - self.last_trigger_time < self.cooldown_seconds:
            logger.debug(f"⏱️ Cooldown active ({self.cooldown_seconds}s)")
            return False

        # Skip reset gesture (fist)
        if finger_count == self.reset_gesture:
            return False

        # Handle toggle gestures (1-5 fingers)
        if 1 <= finger_count <= 5:
            entity_key = f"entity_{finger_count}"
            entity_id = self.entities.get(entity_key, "")

            if entity_id:
                logger.info(f"🎮 {finger_count} finger(s) → Toggling {entity_id}")
                
                # Lock this gesture until hand is removed
                self.gesture_locked = True
                self.locked_gesture = finger_count
                self.last_gesture = finger_count
                self.last_trigger_time = now
                
                await self.toggle_entity(finger_count, entity_id)
                return True
            else:
                logger.warning(f"⚠️ No entity configured for {finger_count} finger(s)")

        return False

    async def disconnect(self):
        """Disconnect from Home Assistant"""
        if self.ws:
            await self.ws.close()
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False


# ========== GESTURE PROCESSOR ==========
class GestureProcessor:
    """Handles RTSP stream reading and gesture detection with stability"""

    def __init__(self, ha_client, sio_server):
        self.ha_client = ha_client
        self.sio_server = sio_server
        self.running = False
        self.thread = None
        self.cap = None

        # Gesture stability - increased buffer for smoothing
        self.gesture_history = deque(maxlen=config["stability_frames"])
        self.current_stable_gesture = None
        self.current_finger_count = 0
        self.hand_detected = False
        
        # Hold time tracking - gesture must be held for minimum time
        self.gesture_hold_start = None
        self.pending_gesture = None
        self.gesture_hold_time = config["gesture_hold_time"]
        
        # Hand tracking for filtering
        self.last_hand_center = None
        self.hand_lost_time = None
        self.hand_lost_threshold = 0.5  # seconds before considering hand "lost"

        # Debug and FPS tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_debug_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # RTSP reconnection tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 30
        self.last_successful_frame = time.time()

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

    def _open_rtsp_stream(self):
        """Open RTSP stream with optimized settings"""
        logger.info(f"🔌 Connecting to RTSP stream: {config['rtsp_url']}")
        
        # Release existing capture if any
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Try different backends
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(config["rtsp_url"], backend)
                
                if self.cap.isOpened():
                    # Set buffer size to minimum for low latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, config["fps"])
                    
                    # Read a test frame
                    ret, _ = self.cap.read()
                    if ret:
                        logger.info(f"✅ RTSP stream connected using backend {backend}")
                        self.consecutive_failures = 0
                        return True
                    
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {e}")
        
        logger.error("❌ Failed to open RTSP stream with any backend")
        return False

    def _select_best_hand(self, multi_hand_landmarks, multi_handedness, frame_width, frame_height):
        """
        Select the best hand to track when multiple hands are detected.
        Prefers: larger hands, more centered hands, right hands
        """
        if not multi_hand_landmarks:
            return None, None
        
        if len(multi_hand_landmarks) == 1:
            handedness = "Right"
            if multi_handedness:
                handedness = multi_handedness[0].classification[0].label
            return multi_hand_landmarks[0].landmark, handedness
        
        best_score = -1
        best_landmarks = None
        best_handedness = "Right"
        
        frame_center_x = 0.5
        frame_center_y = 0.5
        
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            
            # Calculate hand metrics
            hand_size = get_hand_size(landmarks)
            center_x, center_y = get_hand_center(landmarks)
            
            # Skip hands that are too small
            if hand_size < config["min_hand_size"]:
                continue
            
            # Calculate distance from frame center
            center_distance = math.sqrt(
                (center_x - frame_center_x) ** 2 + 
                (center_y - frame_center_y) ** 2
            )
            
            # Score: prefer larger, more centered hands
            # Size contributes positively, distance from center contributes negatively
            score = hand_size * 2 - center_distance
            
            # Bonus for right hand (more common for gestures)
            if multi_handedness and i < len(multi_handedness):
                if multi_handedness[i].classification[0].label == "Right":
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_landmarks = landmarks
                if multi_handedness and i < len(multi_handedness):
                    best_handedness = multi_handedness[i].classification[0].label
        
        return best_landmarks, best_handedness

    def _process_loop(self):
        """Main processing loop with reconnection and stability"""
        global hands
        if hands is None:
            init_mediapipe()

        if not self._open_rtsp_stream():
            self._emit_async("status", {
                "status": "error",
                "message": "Failed to open RTSP stream"
            })
            return

        self._emit_async("status", {
            "status": "connected",
            "message": "Camera connected"
        })

        while self.running:
            try:
                # Check if we need to reconnect
                if self.consecutive_failures > self.max_consecutive_failures:
                    logger.warning("🔄 Too many failures, reconnecting RTSP...")
                    if not self._open_rtsp_stream():
                        time.sleep(5)
                        continue
                    self._emit_async("status", {
                        "status": "reconnected",
                        "message": "Camera reconnected"
                    })

                # Read frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    time.sleep(0.05)
                    continue

                self.consecutive_failures = 0
                self.last_successful_frame = time.time()

                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time

                self.frame_count += 1

                # Periodic debug logging
                if self.frame_count % 100 == 0:
                    logger.info(
                        f"📊 FPS: {self.current_fps}, "
                        f"Hand: {self.hand_detected}, "
                        f"Gesture: {self.current_stable_gesture}, "
                        f"HA: {'✓' if self.ha_client.connected else '✗'}"
                    )

                # Process frame for gesture detection
                self._process_frame(frame)

            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.05)

        if self.cap:
            self.cap.release()
        logger.info("Camera loop ended")

    def _process_frame(self, frame):
        """Process a single frame for gesture detection"""
        try:
            # Convert and flip for mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, 1)

            # Process with MediaPipe
            results = hands.process(rgb_frame)

            current_time = time.time()
            finger_count = -1
            detected_handedness = "Right"
            
            if results.multi_hand_landmarks:
                # Select best hand if multiple detected
                landmarks, detected_handedness = self._select_best_hand(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                    frame.shape[1],
                    frame.shape[0]
                )
                
                if landmarks:
                    self.hand_detected = True
                    self.detection_count += 1
                    self.hand_lost_time = None

                    # Count fingers with improved algorithm
                    raw_count, finger_states = count_extended_fingers(landmarks, detected_handedness)
                    
                    # Apply palm detection correction
                    finger_count = detect_palm_gesture(landmarks, raw_count, finger_states)

                    # Update hand center for tracking
                    self.last_hand_center = get_hand_center(landmarks)
                else:
                    self.hand_detected = False
                    finger_count = -1
            else:
                self.hand_detected = False
                finger_count = -1

            # Handle hand lost/found transitions
            if not self.hand_detected:
                if self.hand_lost_time is None:
                    self.hand_lost_time = current_time
                elif current_time - self.hand_lost_time > self.hand_lost_threshold:
                    # Hand has been gone long enough, unlock gestures
                    if self.ha_client.gesture_locked:
                        self.ha_client.unlock_gesture()
                    self.gesture_history.clear()
                    self.pending_gesture = None
                    self.gesture_hold_start = None

            # Update gesture history
            self.gesture_history.append(finger_count if self.hand_detected else -1)

            # Process gesture stability
            self._update_stable_gesture(current_time)

            # Emit current state to UI
            self._emit_current_state()

        except Exception as e:
            logger.error(f"❌ Gesture detection error: {e}")

    def _update_stable_gesture(self, current_time):
        """Update stable gesture with temporal smoothing and hold time"""
        if len(self.gesture_history) < self.gesture_history.maxlen:
            return

        # Get valid gestures (ignore -1 for no hand)
        valid_gestures = [g for g in self.gesture_history if g >= 0]

        if not valid_gestures:
            # No valid gestures in history
            if self.current_stable_gesture is not None:
                logger.info("👋 Hand lost, clearing gesture")
            self.current_stable_gesture = None
            self.pending_gesture = None
            self.gesture_hold_start = None
            return

        # Find most common gesture with high confidence
        gesture_counts = Counter(valid_gestures)
        most_common_gesture, count = gesture_counts.most_common(1)[0]
        confidence = count / len(self.gesture_history)

        # Require high confidence for stability
        if confidence < 0.7:
            # Low confidence, keep current or reset pending
            self.pending_gesture = None
            self.gesture_hold_start = None
            return

        # Check if this is a new gesture or continuation
        if most_common_gesture != self.pending_gesture:
            # New gesture detected, start hold timer
            self.pending_gesture = most_common_gesture
            self.gesture_hold_start = current_time
            logger.debug(f"🎯 New gesture pending: {most_common_gesture} fingers")
            return

        # Same gesture continuing, check hold time
        hold_duration = current_time - self.gesture_hold_start

        if hold_duration >= self.gesture_hold_time:
            # Gesture held long enough
            if most_common_gesture != self.current_stable_gesture:
                self.current_stable_gesture = most_common_gesture
                self.current_finger_count = most_common_gesture
                
                logger.info(
                    f"🎭 Stable gesture: {most_common_gesture} fingers "
                    f"(confidence: {confidence:.0%}, hold: {hold_duration:.2f}s)"
                )

                # Trigger HA action
                asyncio.run_coroutine_threadsafe(
                    self.ha_client.handle_gesture(most_common_gesture),
                    self.ha_client.loop
                )

                # Emit gesture event
                gesture_info = get_gesture_info(most_common_gesture)
                self._emit_async("gesture", {
                    "fingerCount": most_common_gesture,
                    "gestureName": gesture_info["name"],
                    "gestureIcon": gesture_info["icon"],
                    "handDetected": True,
                    "lastAction": self.ha_client.last_action,
                    "haConnected": self.ha_client.connected,
                    "confidence": confidence,
                    "holdTime": hold_duration,
                    "timestamp": datetime.now().isoformat()
                })

    def _emit_current_state(self):
        """Emit current gesture state to UI"""
        if self.hand_detected and self.current_finger_count >= 0:
            gesture_info = get_gesture_info(self.current_finger_count)
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
            "gestureLocked": self.ha_client.gesture_locked,
            "lockedGesture": self.ha_client.locked_gesture,
            "pendingGesture": self.pending_gesture,
            "timestamp": datetime.now().isoformat(),
            "entityStates": self.ha_client.entity_states,
            "fps": self.current_fps
        })


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
        """Stream video from RTSP source"""
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

        return response

    async def health_check(self, request):
        """Health check endpoint"""
        return aiohttp.web.json_response({
            "status": "ok",
            "ha_connected": self.ha_client.connected if self.ha_client else False,
            "gesture_running": self.gesture_processor.running if self.gesture_processor else False,
            "current_gesture": self.gesture_processor.current_finger_count if self.gesture_processor else 0,
            "hand_detected": self.gesture_processor.hand_detected if self.gesture_processor else False,
            "gesture_locked": self.ha_client.gesture_locked if self.ha_client else False,
            "current_fps": self.gesture_processor.current_fps if self.gesture_processor else 0,
            "uptime": datetime.now().isoformat()
        })

    async def get_config(self, request):
        """Get configuration endpoint"""
        return aiohttp.web.json_response({
            "fps": config["fps"],
            "cooldown_seconds": config["cooldown_seconds"],
            "gesture_hold_time": config["gesture_hold_time"],
            "stability_frames": config["stability_frames"],
            "detection_confidence": config["detection_confidence"],
            "entities": {
                f"entity_{i}": config.get(f"entity_{i}", "")
                for i in range(1, 6)
            }
        })

    async def start(self):
        """Start the server and initialize components"""
        logger.info("=" * 60)
        logger.info("🎮 Starting Gesture Control Add-on v2.1 (Enhanced)")
        logger.info("=" * 60)
        logger.info(f"📹 RTSP: {'✅ Configured' if 'your-camera' not in config['rtsp_url'] else '⚠️ Default'}")
        for i in range(1, 6):
            logger.info(f"💡 Entity {i}: {config.get(f'entity_{i}', 'Not configured')}")
        logger.info(f"⏱️  Cooldown: {config['cooldown_seconds']}s")
        logger.info(f"⏱️  Hold Time: {config['gesture_hold_time']}s")
        logger.info(f"📐 Stability Frames: {config['stability_frames']}")
        logger.info("=" * 60)

        entities = {f"entity_{i}": config.get(f"entity_{i}", "") for i in range(1, 6)}

        self.ha_client = HAClient(
            config["ha_url"],
            config["ha_token"],
            entities,
            config["cooldown_seconds"],
            config["reset_gesture"]
        )
        self.ha_client.set_loop(asyncio.get_event_loop())

        connected = await self.ha_client.connect()
        if not connected:
            logger.error("❌ Failed to connect to HA! Check your token and URL")

        self.gesture_processor = GestureProcessor(self.ha_client, self.sio)
        self.gesture_processor.set_loop(asyncio.get_event_loop())
        self.gesture_processor.start()

        self.ha_keepalive_task = asyncio.create_task(self._ha_keepalive())

        runner = aiohttp.web.AppRunner(self.app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "0.0.0.0", 8099)
        await site.start()

        logger.info("🌐 Web server started on port 8099")
        logger.info("📍 UI: http://[YOUR-IP]:8099/")
        logger.info("=" * 60)

        await asyncio.Event().wait()

    async def _ha_keepalive(self):
        """Keep HA connection alive"""
        while True:
            await asyncio.sleep(30)
            if self.ha_client and not self.ha_client.connected:
                logger.info("🔄 Reconnecting to HA...")
                await self.ha_client.connect()


# ========== MAIN ==========
def main():
    """Main entry point"""
    server = GestureServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        if hasattr(server, 'gesture_processor') and server.gesture_processor:
            server.gesture_processor.stop()
        if hasattr(server, 'ha_client') and server.ha_client:
            asyncio.run(server.ha_client.disconnect())


if __name__ == "__main__":
    main()
