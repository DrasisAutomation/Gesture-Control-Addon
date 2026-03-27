#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
Supports 5 entities with 1-5 finger gestures
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
            "gesture_mappings": [
                {"finger_count": 1, "entity_id": "light.entity_1", "name": "Entity 1"},
                {"finger_count": 2, "entity_id": "light.entity_2", "name": "Entity 2"},
                {"finger_count": 3, "entity_id": "light.entity_3", "name": "Entity 3"},
                {"finger_count": 4, "entity_id": "light.entity_4", "name": "Entity 4"},
                {"finger_count": 5, "entity_id": "light.entity_5", "name": "Entity 5"}
            ],
            "reset_gesture": 0,  # Fist resets tracking
            "fps": 15,
            "frame_width": 640,
            "frame_height": 480,
            "cooldown_seconds": 1.5,
            "detection_confidence": 0.6,
            "tracking_confidence": 0.6,
            "hand_stability_frames": 3  # Frames needed for stable gesture
        }
    
    # Ensure all keys exist
    defaults = {
        "fps": 15,
        "frame_width": 640,
        "frame_height": 480,
        "cooldown_seconds": 1.5,
        "detection_confidence": 0.6,
        "tracking_confidence": 0.6,
        "hand_stability_frames": 3,
        "reset_gesture": 0
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
mp_drawing = mp.solutions.drawing_utils
hands = None

def init_mediapipe():
    """Initialize MediaPipe Hands"""
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config["detection_confidence"],
        min_tracking_confidence=config["tracking_confidence"],
        model_complexity=0  # Use 0 for better performance
    )
    logger.info("✅ MediaPipe initialized")

def count_extended_fingers(landmarks, image_width, image_height):
    """
    Improved finger counting with better thumb detection
    Returns number of extended fingers (1-5)
    """
    if not landmarks:
        return 0
    
    # Get fingertip and pip joint coordinates
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
    
    count = 0
    
    # For fingers (index, middle, ring, pinky)
    for i in range(1, 5):  # Skip thumb
        tip_y = landmarks[fingertips[i]].y
        pip_y = landmarks[finger_pips[i]].y
        
        # Finger is extended if tip is above PIP (y is smaller)
        if tip_y < pip_y - 0.02:
            count += 1
    
    # Thumb detection (based on x-coordinate and orientation)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    hand_orientation = landmarks[5].x  # Reference point
    
    # Check if thumb is extended
    if hand_orientation < 0.5:  # Right hand orientation
        if thumb_tip.x > thumb_ip.x + 0.03:
            count += 1
    else:  # Left hand orientation
        if thumb_tip.x < thumb_ip.x - 0.03:
            count += 1
    
    return count

def get_gesture_info(finger_count):
    """Return gesture name and icon for finger count"""
    mapping = {
        0: {"name": "Reset/Fist", "icon": "✊", "description": "Reset tracking"},
        1: {"name": "1 Finger", "icon": "☝️", "description": "Toggle Entity 1"},
        2: {"name": "2 Fingers", "icon": "✌️", "description": "Toggle Entity 2"},
        3: {"name": "3 Fingers", "icon": "👌", "description": "Toggle Entity 3"},
        4: {"name": "4 Fingers", "icon": "🖖", "description": "Toggle Entity 4"},
        5: {"name": "5 Fingers/Palm", "icon": "✋", "description": "Toggle Entity 5"}
    }
    return mapping.get(finger_count, {"name": f"{finger_count} Fingers", "icon": "🖐️", "description": "Unknown"})

# ========== HOME ASSISTANT WEBSOCKET ==========
class HAClient:
    """Home Assistant WebSocket Client"""
    
    def __init__(self, url, token, gesture_mappings, cooldown_seconds, reset_gesture=0):
        self.url = url
        self.token = token
        self.gesture_mappings = {m["finger_count"]: m for m in gesture_mappings}
        self.cooldown_seconds = cooldown_seconds
        self.reset_gesture = reset_gesture
        
        self.ws = None
        self.session = None
        self.connected = False
        self.last_trigger_time = {}
        self.last_action = ""
        self.loop = None
        
        # Track last triggered gesture to prevent rapid toggles
        self.last_triggered_gesture = None
        self.last_trigger_timestamp = 0
        
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
            if msg.get("type") != "auth_required":
                logger.error(f"❌ Expected auth_required, got: {msg}")
                self.connected = False
                return False
            
            auth_msg = {"type": "auth", "access_token": self.token}
            await self.ws.send_json(auth_msg)
            
            msg = await self.ws.receive_json()
            
            if msg.get("type") == "auth_ok":
                self.connected = True
                logger.info("✅ HA WebSocket connected and authenticated")
                return True
            else:
                logger.error(f"❌ HA auth failed: {msg}")
                self.connected = False
                return False
                
        except Exception as e:
            logger.error(f"❌ HA connection error: {e}")
            self.connected = False
            return False
    
    async def toggle_entity(self, entity_id):
        """Toggle an entity (works for lights, switches, etc.)"""
        if not self.connected or not self.ws:
            logger.warning("⚠️ Not connected to HA, cannot toggle entity")
            return False
        
        try:
            domain = entity_id.split('.')[0]
            msg_id = int(time.time() * 1000)
            call_msg = {
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": "toggle",
                "service_data": {"entity_id": entity_id}
            }
            await self.ws.send_json(call_msg)
            logger.info(f"⚡ Toggled {entity_id}")
            self.last_action = f"Toggle {entity_id.split('.')[-1]}"
            return True
        except Exception as e:
            logger.error(f"❌ Failed to toggle entity: {e}")
            self.connected = False
            return False
    
    async def handle_gesture(self, finger_count):
        """Handle gesture and trigger HA actions"""
        if not self.connected:
            logger.debug(f"⚠️ Not connected to HA, ignoring gesture: {finger_count}")
            return False
        
        now = time.time()
        
        # Check global cooldown for same gesture
        if self.last_triggered_gesture == finger_count and (now - self.last_trigger_timestamp) < self.cooldown_seconds:
            logger.debug(f"⏱️ Cooldown active for gesture: {finger_count}")
            return False
        
        # Handle reset gesture (fist)
        if finger_count == self.reset_gesture:
            logger.info(f"🔄 Reset gesture detected - clearing tracking")
            self.last_triggered_gesture = None
            self.last_trigger_timestamp = now
            return True
        
        # Handle entity toggle
        if finger_count in self.gesture_mappings:
            entity_info = self.gesture_mappings[finger_count]
            entity_id = entity_info["entity_id"]
            
            # Check per-entity cooldown
            last_trigger = self.last_trigger_time.get(entity_id, 0)
            if now - last_trigger < self.cooldown_seconds:
                logger.debug(f"⏱️ Cooldown active for {entity_id}")
                return False
            
            logger.info(f"🎮 Toggling {entity_id} (finger count: {finger_count})")
            await self.toggle_entity(entity_id)
            
            self.last_trigger_time[entity_id] = now
            self.last_triggered_gesture = finger_count
            self.last_trigger_timestamp = now
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
        
        # Gesture stability with sliding window
        self.gesture_history = deque(maxlen=config["hand_stability_frames"])
        self.current_stable_gesture = 0
        self.current_finger_count = 0
        self.hand_detected = False
        self.hand_confidence = 0
        
        # Performance monitoring
        self.frame_count = 0
        self.detection_count = 0
        self.last_debug_time = time.time()
        self.processing_time = 0
        
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
            self.thread.join(timeout=3)
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
    
    def _get_stable_gesture(self):
        """Get the most common gesture from history, ignoring no-hand"""
        valid_gestures = [g for g in self.gesture_history if g >= 0]
        if not valid_gestures:
            return -1
        
        from collections import Counter
        return Counter(valid_gestures).most_common(1)[0][0]
    
    def _process_loop(self):
        """Main processing loop in separate thread"""
        global hands
        if hands is None:
            init_mediapipe()
        
        # Open RTSP stream with retry logic
        max_retries = 3
        retry_count = 0
        
        while self.running and retry_count < max_retries:
            logger.info(f"🔌 Connecting to RTSP stream (attempt {retry_count + 1})...")
            
            self.cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open RTSP stream (attempt {retry_count + 1})")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(3)
                continue
            
            # Set properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, config["fps"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
            
            logger.info("✅ RTSP stream connected successfully")
            self._emit_async("status", {
                "status": "connected",
                "message": "Camera connected"
            })
            
            # Reset retry counter on successful connection
            retry_count = 0
            
            # Frame processing loop
            frame_skip = 0
            process_every_n = max(1, 30 // config["fps"])
            last_frame_time = time.time()
            
            while self.running:
                try:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("⚠️ Failed to read frame, reconnecting...")
                        break
                    
                    self.frame_count += 1
                    current_time = time.time()
                    
                    # Calculate actual FPS
                    if self.frame_count % 30 == 0:
                        fps = 30 / (current_time - last_frame_time)
                        logger.debug(f"📊 FPS: {fps:.1f}")
                        last_frame_time = current_time
                    
                    # Process every Nth frame
                    frame_skip += 1
                    if frame_skip >= process_every_n:
                        frame_skip = 0
                        process_start = time.time()
                        
                        try:
                            # Convert to RGB and flip horizontally for mirror effect
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            rgb_frame = cv2.flip(rgb_frame, 1)
                            
                            # Process with MediaPipe
                            results = hands.process(rgb_frame)
                            
                            finger_count = 0
                            self.hand_detected = False
                            hand_landmarks = None
                            
                            if results.multi_hand_landmarks:
                                self.hand_detected = True
                                self.detection_count += 1
                                
                                # Get first hand
                                hand_landmarks = results.multi_hand_landmarks[0]
                                landmarks = hand_landmarks.landmark
                                finger_count = count_extended_fingers(landmarks, 
                                                                      frame.shape[1], 
                                                                      frame.shape[0])
                                
                                # Log detection periodically
                                if self.detection_count % 30 == 0:
                                    logger.info(f"✋ Hand detected - {finger_count} fingers")
                            
                            # Update gesture history
                            if self.hand_detected:
                                self.gesture_history.append(finger_count)
                            else:
                                self.gesture_history.append(-1)  # No hand
                            
                            # Get stable gesture
                            stable_gesture = self._get_stable_gesture()
                            
                            # Check if stable gesture changed
                            if stable_gesture >= 0 and stable_gesture != self.current_stable_gesture:
                                self.current_stable_gesture = stable_gesture
                                logger.info(f"🎭 Stable gesture detected: {stable_gesture} fingers")
                                
                                # Trigger HA action in async thread
                                asyncio.run_coroutine_threadsafe(
                                    self.ha_client.handle_gesture(stable_gesture),
                                    self.ha_client.loop
                                )
                            
                            self.current_finger_count = stable_gesture if stable_gesture >= 0 else 0
                            
                            # Calculate processing time
                            self.processing_time = (time.time() - process_start) * 1000
                            
                            # Broadcast gesture data
                            gesture_info = get_gesture_info(self.current_finger_count)
                            
                            # Get mapped entity info
                            mapped_entity = None
                            if self.current_finger_count in self.ha_client.gesture_mappings:
                                mapped_entity = self.ha_client.gesture_mappings[self.current_finger_count]["name"]
                            
                            self._emit_async("gesture", {
                                "fingerCount": self.current_finger_count,
                                "gestureName": gesture_info["name"],
                                "gestureIcon": gesture_info["icon"],
                                "handDetected": self.hand_detected,
                                "lastAction": self.ha_client.last_action,
                                "haConnected": self.ha_client.connected,
                                "mappedEntity": mapped_entity,
                                "processingTime": round(self.processing_time, 1),
                                "timestamp": datetime.now().isoformat()
                            })
                            
                        except Exception as e:
                            logger.error(f"❌ Gesture detection error: {e}")
                    
                except Exception as e:
                    logger.error(f"❌ Frame processing error: {e}")
                    time.sleep(0.05)
            
            # Release camera on disconnect
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if self.running and retry_count < max_retries:
                logger.info("🔄 Reconnecting to RTSP stream...")
                time.sleep(2)
        
        logger.error("❌ Failed to connect to RTSP stream after multiple attempts")
        self._emit_async("status", {
            "status": "error",
            "message": "Failed to connect to camera stream"
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
                "gestureMappings": config["gesture_mappings"],
                "message": "Connected to server"
            })
        
        @self.sio.on("disconnect")
        async def handle_disconnect(sid):
            logger.info(f"🔴 Frontend disconnected: {sid}")
    
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
    
    async def get_config(self, request):
        """Get current configuration"""
        return aiohttp.web.json_response({
            "gesture_mappings": config["gesture_mappings"],
            "rtsp_url": config["rtsp_url"],
            "fps": config["fps"],
            "cooldown_seconds": config["cooldown_seconds"],
            "reset_gesture": config["reset_gesture"],
            "ha_connected": self.ha_client.connected if self.ha_client else False
        })
    
    async def health_check(self, request):
        """Health check endpoint"""
        return aiohttp.web.json_response({
            "status": "ok",
            "ha_connected": self.ha_client.connected if self.ha_client else False,
            "gesture_running": self.gesture_processor.running if self.gesture_processor else False,
            "current_gesture": self.gesture_processor.current_finger_count if self.gesture_processor else 0,
            "hand_detected": self.gesture_processor.hand_detected if self.gesture_processor else False,
            "processing_time": self.gesture_processor.processing_time if self.gesture_processor else 0,
            "fps": config["fps"],
            "uptime": datetime.now().isoformat()
        })
    
    async def start(self):
        """Start the server and initialize components"""
        # Print startup banner
        logger.info("=" * 60)
        logger.info("🎮 Starting Gesture Control Add-on (Enhanced)")
        logger.info("=" * 60)
        logger.info(f"📹 RTSP URL: {config['rtsp_url']}")
        logger.info(f"🎯 Gesture Mappings:")
        for mapping in config["gesture_mappings"]:
            logger.info(f"   - {mapping['finger_count']} fingers → {mapping['entity_id']} ({mapping['name']})")
        logger.info(f"🔄 Reset gesture: {config['reset_gesture']} fingers (Fist)")
        logger.info(f"⏱️  Cooldown: {config['cooldown_seconds']}s")
        logger.info(f"🎯 Detection Confidence: {config['detection_confidence']}")
        logger.info(f"🎯 Tracking Confidence: {config['tracking_confidence']}")
        logger.info(f"📐 Resolution: {config['frame_width']}x{config['frame_height']}")
        logger.info(f"⚡ Target FPS: {config['fps']}")
        logger.info(f"🔌 HA URL: {config['ha_url']}")
        logger.info("=" * 60)
        
        # Initialize HA client
        self.ha_client = HAClient(
            config["ha_url"],
            config["ha_token"],
            config["gesture_mappings"],
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