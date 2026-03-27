#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant
Single-file implementation with RTSP streaming and gesture detection
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
            "strip_light_entity": "light.strip_light",
            "row_light_entity": "light.row_light",
            "fps": 15,
            "frame_width": 640,
            "frame_height": 480,
            "bitrate": "500k",
            "cooldown_seconds": 2,
            "detection_confidence": 0.5,
            "tracking_confidence": 0.5,
        }
    
    # Ensure all keys exist
    defaults = {
        "fps": 15,
        "frame_width": 640,
        "frame_height": 480,
        "bitrate": "500k",
        "cooldown_seconds": 2,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
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
        max_num_hands=1,
        min_detection_confidence=config["detection_confidence"],
        min_tracking_confidence=config["tracking_confidence"],
        model_complexity=0  # Use simpler model for better performance
    )
    logger.info("✅ MediaPipe initialized")

def count_extended_fingers(landmarks, image_width, image_height):
    """
    Count number of extended fingers from hand landmarks
    Returns count 0-5
    """
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
    
    # Thumb based on hand orientation
    is_right_hand = landmarks[5].x < landmarks[17].x
    if is_right_hand:
        if landmarks[4].x < landmarks[3].x - 0.02:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x + 0.02:
            count += 1
    
    return count

def get_gesture_info(finger_count):
    """Return gesture name and icon for finger count"""
    mapping = {
        0: {"name": "Fist", "icon": "✊"},
        2: {"name": "2 Fingers", "icon": "✌️"},
        3: {"name": "3 Fingers", "icon": "👌"},
        5: {"name": "Palm", "icon": "✋"}
    }
    if finger_count in mapping:
        return mapping[finger_count]
    return {"name": f"{finger_count} Fingers", "icon": "🖐️"}

# ========== HOME ASSISTANT WEBSOCKET ==========
class HAClient:
    """Home Assistant WebSocket Client"""
    
    def __init__(self, url, token, strip_entity, row_entity, cooldown_seconds):
        self.url = url
        self.token = token
        self.strip_entity = strip_entity
        self.row_entity = row_entity
        self.cooldown_seconds = cooldown_seconds
        
        self.ws = None
        self.session = None
        self.connected = False
        self.last_gesture = None
        self.last_trigger_time = 0
        self.last_action = ""
        self.loop = None
        self.reconnect_task = None
        
    def set_loop(self, loop):
        self.loop = loop
        
    async def connect(self):
        """Connect to Home Assistant WebSocket"""
        try:
            # Close existing session if any
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(self.url)
            
            # Send auth
            await self.ws.send_json({
                "type": "auth",
                "access_token": self.token
            })
            
            # Wait for auth response
            msg = await self.ws.receive_json()
            if msg.get("type") == "auth_ok":
                self.connected = True
                logger.info("✅ HA WebSocket connected and authenticated")
                return True
            elif msg.get("type") == "auth_invalid":
                logger.error(f"❌ HA auth invalid: {msg.get('message', 'Invalid token')}")
                logger.error("Please check your HA token in add-on configuration")
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
    
    async def call_service(self, domain, service, entity):
        """Call a service on Home Assistant"""
        if not self.connected or not self.ws:
            logger.warning("⚠️ Not connected to HA, cannot call service")
            return False
        
        try:
            msg_id = int(time.time() * 1000)
            await self.ws.send_json({
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "service_data": {"entity_id": entity}
            })
            logger.info(f"⚡ HA call: {service} {entity}")
            self.last_action = f"{service} {entity.split('.')[-1]}"
            return True
        except Exception as e:
            logger.error(f"❌ Failed to call service: {e}")
            self.connected = False
            return False
    
    async def handle_gesture(self, finger_count):
        """Handle gesture and trigger HA actions with cooldown"""
        if not self.connected:
            logger.debug("Not connected to HA, ignoring gesture")
            return False
            
        now = time.time()
        
        # Cooldown check
        if (self.last_gesture == finger_count and 
            now - self.last_trigger_time < self.cooldown_seconds):
            return False
        
        action = None
        
        # Map finger count to action
        if finger_count == 0:
            action = ("light", "turn_on", self.strip_entity)
        elif finger_count == 5:
            action = ("light", "turn_off", self.strip_entity)
        elif finger_count == 2:
            action = ("light", "turn_on", self.row_entity)
        elif finger_count == 3:
            action = ("light", "turn_off", self.row_entity)
        
        if action:
            self.last_gesture = finger_count
            self.last_trigger_time = now
            domain, service, entity = action
            await self.call_service(domain, service, entity)
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
        
        # Gesture stability
        self.gesture_history = deque(maxlen=5)
        self.current_stable_gesture = None
        self.current_finger_count = 0
        self.hand_detected = False
        
        # Debug counters
        self.frame_count = 0
        self.detection_count = 0
        self.last_debug_time = time.time()
        
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
        # Initialize MediaPipe in this thread
        global hands
        if hands is None:
            init_mediapipe()
        
        # Open RTSP stream
        logger.info(f"🔌 Connecting to RTSP stream...")
        
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
        
        frame_skip = 0
        process_every_n = max(1, 30 // config["fps"])
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Debug every 100 frames
                if self.frame_count % 100 == 0:
                    fps = self.frame_count / (time.time() - self.last_debug_time)
                    logger.info(f"📊 Stats: {self.frame_count} frames, {self.detection_count} detections, {fps:.1f} FPS")
                    self.last_debug_time = time.time()
                    self.frame_count = 0
                    self.detection_count = 0
                
                # Process every Nth frame
                frame_skip += 1
                if frame_skip >= process_every_n:
                    frame_skip = 0
                    
                    try:
                        # Convert to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process with MediaPipe
                        results = hands.process(rgb_frame)
                        
                        finger_count = 0
                        self.hand_detected = False
                        
                        if results.multi_hand_landmarks:
                            self.hand_detected = True
                            self.detection_count += 1
                            
                            # Get first hand
                            landmarks = results.multi_hand_landmarks[0].landmark
                            finger_count = count_extended_fingers(landmarks, 
                                                                  frame.shape[1], 
                                                                  frame.shape[0])
                            
                            # Log detection occasionally
                            if self.detection_count % 30 == 0:
                                logger.info(f"✋ Hand detected - {finger_count} fingers")
                        
                        # Update gesture history (only count if hand detected)
                        if self.hand_detected:
                            self.gesture_history.append(finger_count)
                        else:
                            self.gesture_history.append(-1)  # No hand
                        
                        # Find stable gesture
                        if len(self.gesture_history) == self.gesture_history.maxlen:
                            # Count only hand detections (ignore -1)
                            valid_gestures = [g for g in self.gesture_history if g >= 0]
                            if valid_gestures:
                                from collections import Counter
                                stable = Counter(valid_gestures).most_common(1)[0][0]
                                
                                if stable != self.current_stable_gesture:
                                    self.current_stable_gesture = stable
                                    logger.info(f"🎭 Stable gesture: {stable} fingers")
                                    
                                    # Trigger HA action
                                    asyncio.run_coroutine_threadsafe(
                                        self.ha_client.handle_gesture(stable),
                                        self.ha_client.loop
                                    )
                                
                                self.current_finger_count = stable
                            else:
                                self.current_finger_count = 0
                                self.current_stable_gesture = None
                        else:
                            self.current_finger_count = finger_count if self.hand_detected else 0
                        
                        # Broadcast gesture data
                        if self.hand_detected:
                            gesture_info = get_gesture_info(self.current_finger_count)
                            gesture_name = gesture_info["name"]
                            gesture_icon = gesture_info["icon"]
                        else:
                            gesture_name = "No Hand"
                            gesture_icon = "🤚"
                        
                        self._emit_async("gesture", {
                            "fingerCount": self.current_finger_count if self.hand_detected else 0,
                            "gestureName": gesture_name,
                            "gestureIcon": gesture_icon,
                            "handDetected": self.hand_detected,
                            "lastAction": self.ha_client.last_action,
                            "haConnected": self.ha_client.connected,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Gesture detection error: {e}")
                
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
                time.sleep(0.1)
        
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
            "gesture_running": self.gesture_processor.running if self.gesture_processor else False,
            "current_gesture": self.gesture_processor.current_finger_count if self.gesture_processor else 0,
            "hand_detected": self.gesture_processor.hand_detected if self.gesture_processor else False,
            "fps": config["fps"],
            "uptime": datetime.now().isoformat()
        })
    
    async def start(self):
        """Start the server and initialize components"""
        # Print startup banner
        logger.info("=" * 60)
        logger.info("🎮 Starting Gesture Control Add-on")
        logger.info("=" * 60)
        logger.info(f"📹 RTSP URL: {'✅ Configured' if config['rtsp_url'] != 'rtsp://your-camera-ip:554/stream' else '⚠️ Using default'}")
        logger.info(f"💡 Strip Light: {config['strip_light_entity']}")
        logger.info(f"💡 Row Light: {config['row_light_entity']}")
        logger.info(f"⏱️  Cooldown: {config['cooldown_seconds']}s")
        logger.info(f"🎯 Detection Confidence: {config['detection_confidence']}")
        logger.info(f"🎯 Tracking Confidence: {config['tracking_confidence']}")
        logger.info(f"📐 Resolution: {config['frame_width']}x{config['frame_height']}")
        logger.info(f"⚡ Target FPS: {config['fps']}")
        logger.info("=" * 60)
        
        # Initialize HA client
        self.ha_client = HAClient(
            config["ha_url"],
            config["ha_token"],
            config["strip_light_entity"],
            config["row_light_entity"],
            config["cooldown_seconds"]
        )
        self.ha_client.set_loop(asyncio.get_event_loop())
        
        # Connect to HA
        await self.ha_client.connect()
        
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