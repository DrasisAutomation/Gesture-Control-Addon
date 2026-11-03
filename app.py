#!/usr/bin/env python3
"""
Gesture Control Add-on for Home Assistant with ChatGPT AI Verification
Prevents false-positive triggers caused by phone calls, holding objects, or resting postures.
"""

import asyncio
import base64
from collections import Counter, deque
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiohttp.web
import cv2
import httpx
import mediapipe as mp
import numpy as np
import socketio

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("gesture-ai-addon")

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
HA_OPTIONS_PATH = Path("/data/options.json")


def load_env_file() -> Dict[str, str]:
    """Parse key-value pairs from .env if present."""
    data = {}
    if ENV_PATH.exists():
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        data[k.strip().lower()] = v.strip()
        except Exception as e:
            logger.warning(f"Failed to read .env: {e}")
    return data


def load_config() -> Dict[str, Any]:
    """Load configuration from Home Assistant options.json or .env fallback."""
    config: Dict[str, Any] = {}

    # Check HA add-on options
    if HA_OPTIONS_PATH.exists():
        try:
            with open(HA_OPTIONS_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("✅ Loaded configuration from /data/options.json")
        except Exception as e:
            logger.warning(f"Error reading /data/options.json: {e}")

    # Fallback to .env
    env_vars = load_env_file()

    defaults: Dict[str, Any] = {
        "openai_api_key": env_vars.get("openai_api_key", ""),
        "openai_model": env_vars.get("openai_model", "gpt-4o-mini"),
        "ai_verification_enabled": env_vars.get("ai_verification_enabled", "true").lower() == "true",
        "rtsp_url": env_vars.get("rtsp_url", ""),
        "camera_source": env_vars.get("camera_source", "0"),
        "ha_url": env_vars.get("ha_url", "https://demo.lumihomepro1.com/"),
        "ha_token": env_vars.get("ha_token", ""),
        "entity_1": env_vars.get("entity_1", "light.pantry_light1"),
        "entity_2": env_vars.get("entity_2", "light.pantry_light2"),
        "entity_3": env_vars.get("entity_3", "light.pantry_light3"),
        "entity_4": env_vars.get("entity_4", "light.pantry_light4"),
        "entity_5": env_vars.get("entity_5", "light.pantry_light6"),
        "fps": int(env_vars.get("fps", 20)),
        "frame_width": int(env_vars.get("frame_width", 640)),
        "frame_height": int(env_vars.get("frame_height", 480)),
        "cooldown_seconds": float(env_vars.get("cooldown_seconds", 1.5)),
        "capture_delay_seconds": float(env_vars.get("capture_delay_seconds", 1.0)),
        "detection_confidence": float(env_vars.get("detection_confidence", 0.55)),
        "tracking_confidence": float(env_vars.get("tracking_confidence", 0.55)),
        "reset_gesture": int(env_vars.get("reset_gesture", 0)),
        "stability_frames": int(env_vars.get("stability_frames", 6)),
        "port": int(env_vars.get("port", 8099)),
    }

    for key, val in defaults.items():
        if key not in config or config[key] == "" or config[key] is None:
            config[key] = val

    return config


config = load_config()


def save_config_to_env(new_config: Dict[str, Any]):
    """Persist updated config back to .env."""
    lines = [
        "# Drasis Gesture Control Add-on Configuration",
        f"OPENAI_API_KEY={new_config.get('openai_api_key', '')}",
        f"OPENAI_MODEL={new_config.get('openai_model', 'gpt-4o-mini')}",
        f"AI_VERIFICATION_ENABLED={'true' if new_config.get('ai_verification_enabled', True) else 'false'}",
        f"HA_URL={new_config.get('ha_url', '')}",
        f"HA_TOKEN={new_config.get('ha_token', '')}",
        f"CAMERA_SOURCE={new_config.get('camera_source', '0')}",
        f"RTSP_URL={new_config.get('rtsp_url', '')}",
        f"FRAME_WIDTH={new_config.get('frame_width', 640)}",
        f"FRAME_HEIGHT={new_config.get('frame_height', 480)}",
        f"FPS={new_config.get('fps', 20)}",
        f"STABILITY_FRAMES={new_config.get('stability_frames', 6)}",
        f"COOLDOWN_SECONDS={new_config.get('cooldown_seconds', 1.5)}",
        f"CAPTURE_DELAY_SECONDS={new_config.get('capture_delay_seconds', 1.0)}",
        f"DETECTION_CONFIDENCE={new_config.get('detection_confidence', 0.55)}",
        f"TRACKING_CONFIDENCE={new_config.get('tracking_confidence', 0.55)}",
        f"RESET_GESTURE={new_config.get('reset_gesture', 0)}",
        f"ENTITY_1={new_config.get('entity_1', '')}",
        f"ENTITY_2={new_config.get('entity_2', '')}",
        f"ENTITY_3={new_config.get('entity_3', '')}",
        f"ENTITY_4={new_config.get('entity_4', '')}",
        f"ENTITY_5={new_config.get('entity_5', '')}",
        f"PORT={new_config.get('port', 8099)}",
    ]
    try:
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Saved settings to .env")
    except Exception as e:
        logger.error(f"Failed to save .env: {e}")


# ========== LIGHTING ENHANCEMENT FOR LOW-LIGHT & BACKLIT SCENES ==========
def enhance_lighting(frame_bgr: np.ndarray) -> np.ndarray:
    """Enhance shadows and contrast for dim lighting and backlit scenes so fingers are easily detected."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = float(np.mean(l))

    # If lighting is dim (mean_l < 125) or backlit (strong background light casting shadows on hand)
    if mean_l < 125:
        # Gamma shadow lift (< 1.0 brightens deep shadows without blowing out highlights)
        gamma = 0.55 if mean_l < 60 else 0.72
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_boosted = cv2.LUT(l, lut)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) brings out finger contours & knuckles
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_boosted)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    return frame_bgr


# ========== DUAL-VIEW HIGH-RESOLUTION COMPOSITE FOR CHATGPT VISION ==========
def make_dual_view(
    frame_bgr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    other_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
) -> np.ndarray:
    """Create a high-resolution side-by-side composite:
    - Left: Full scene showing user posture, distance, and context with all detected hands (480x360)
    - Right: High-resolution zoomed inspection crop of the active candidate hand with shadow lift (360x360)
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # Draw all other detected hands in full scene context (subtle slate/amber boxes)
    if other_bboxes:
        for ob in other_bboxes:
            if ob and ob != bbox:
                cv2.rectangle(annotated, (ob[0], ob[1]), (ob[2], ob[3]), (200, 160, 50), 1)

    # Highlight active candidate hand in prominent yellow/cyan
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)

    # 1. Full scene view (scaled to 480x360)
    scene_h, scene_w = 360, 480
    scene_view = cv2.resize(annotated, (scene_w, scene_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(scene_view, (0, 0), (scene_w, 28), (15, 23, 42), -1)
    cv2.putText(scene_view, "1. FULL SCENE CONTEXT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    # 2. Zoomed Hand Inspection Crop (360x360) from clean unannotated frame
    crop_size = 360
    if bbox:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * 0.35)
        pad_y = int(bh * 0.35)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        hand_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if hand_crop.size > 0:
            hand_crop_enh = enhance_lighting(hand_crop)
            zoomed_hand = cv2.resize(hand_crop_enh, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            zoomed_hand = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    else:
        zoomed_hand = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    cv2.rectangle(zoomed_hand, (0, 0), (crop_size, 28), (15, 23, 42), -1)
    cv2.putText(zoomed_hand, "2. ZOOMED HAND INSPECTION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    # Horizontal concatenation: 480 + 360 = 840 wide, 360 tall
    composite = np.hstack((scene_view, zoomed_hand))
    return composite


# ========== CHATGPT VISION AI VERIFICATION ENGINE ==========
CHATGPT_SYSTEM_PROMPT = """You are an expert AI vision verifier for smart home gesture control.
MediaPipe candidate detection: {gesture_name} ({finger_count} fingers).

You are inspecting a composite image:
- LEFT: Full scene context showing the person's face, eyes/gaze, body posture, and all detected hands (active candidate hand in yellow box).
- RIGHT: Zoomed high-resolution inspection of the active candidate hand.

========================================================================
MANDATORY TWO-FACTOR VALIDATION (BOTH MUST BE TRUE TO EXECUTE):
========================================================================
To verify a gesture as valid (is_valid_gesture=true), BOTH factors MUST be satisfied. If either factor fails, you MUST reject (is_valid_gesture=false):

FACTOR 1: PERSON'S FACE IS IN FRONT OF / ORIENTED TOWARD THE CAMERA
- Look at the person making the gesture in the LEFT panel (full scene view).
- CRITICAL USER RULE:
  * Direct eye pupil tracking or staring directly into the camera lens hole is NOT REQUIRED! Smart home and security cameras are often mounted high up on a wall or ceiling.
  * AS LONG AS THE PERSON'S FACE IS VISIBLE IN FRONT OF THE CAMERA AND ORIENTED GENERALLY FORWARD, THAT IS FULLY VALID AND ENOUGH!
- If the person's face is in front of the camera and facing forward toward the camera/room, set person_facing_camera=true.
- ONLY reject (person_facing_camera=false) IF:
  * The person has their back completely turned to the camera.
  * The person's head is turned sideways (90+ degrees away) talking to someone behind or beside them.
  * The person's head is tilted far down looking directly into their hands/lap.
  * There is no person visible in front of the camera.

FACTOR 2: HAND IS INTENTIONALLY PRESENTING A COMMAND GESTURE TO CAMERA
- The hand highlighted in the yellow box is DELIBERATELY RAISED AND SHOWN TO THE CAMERA as an intentional smart home command.
- The inside palm is facing forward toward the camera with 1, 2, 3, 4, or 5 fingers clearly and uprightly extended in the air.
- If the user holds up their hand showing open fingers toward the camera, THIS IS AN INTENTIONAL COMMAND GESTURE! Set hand_intentionally_showing_gesture=true.
- ONLY reject IF:
  * Casual motion: Actively scratching head, rubbing eye/face, touching hair, or resting chin on hand.
  * Holding object: Hand is holding a phone, remote, mug, bottle, pen, or item.
  * Passive resting: Hand is resting flat on a desk, keyboard, armrest, or lap.
  * Casual conversation: Gesturing with hands while talking to another person with head turned away.

========================================================================
FINGER COUNTING ACCURACY (ZOOMED VIEW ON RIGHT):
========================================================================
Carefully inspect the ZOOMED HAND on the right and count the EXACT extended fingers:
- 1: 1 Finger (Index upright ☝️)
- 2: 2 Fingers (Peace sign ✌️: Index+Middle, or L-shape: Thumb+Index)
- 3: 3 Fingers (Index+Middle+Ring 👌, or Thumb+Index+Middle)
- 4: 4 FINGERS (All 4 main fingers - Index, Middle, Ring, Pinky - upright with THUMB FOLDED INTO PALM)
- 5: 5 FINGERS (Open palm ✋: All 4 main fingers upright PLUS thumb extended outward)

4 VS 5 FINGERS RULE:
- If 4 fingers are upright and the THUMB IS FOLDED/CURLED INTO THE PALM, you MUST verify as 4 Fingers!
- Only verify as 5 fingers if the thumb is visibly extended outward to the side away from the palm.

========================================================================
FINAL VERDICT & EXECUTION DECISION:
========================================================================
- If FACTOR 1 (Face facing camera) IS TRUE AND FACTOR 2 (Intentional hand gesture) IS TRUE:
  --> is_valid_gesture = true, is_false_trigger = false
- If EITHER factor is FALSE:
  --> is_valid_gesture = false, is_false_trigger = true

Respond ONLY with valid JSON:
If valid (will execute):
{
  "is_valid_gesture": true,
  "is_false_trigger": false,
  "person_facing_camera": true,
  "hand_intentionally_showing_gesture": true,
  "gesture": "{gesture_name}",
  "verified_finger_count": {finger_count},
  "confidence": 0.98,
  "rejection_reason": "None",
  "explanation": "Person is in front of camera facing forward and intentionally presenting a clear command gesture."
}

If false trigger (will NOT execute):
{
  "is_valid_gesture": false,
  "is_false_trigger": true,
  "person_facing_camera": false,
  "hand_intentionally_showing_gesture": false,
  "gesture": "None",
  "verified_finger_count": 0,
  "confidence": 0.95,
  "rejection_reason": "<rejection_reason>",
  "explanation": "<concise explanation of why rejected>"
}
"""


class ChatGPTVerifier:
    """Verifies candidate gestures using OpenAI gpt-4o-mini Vision to reject false triggers."""

    def __init__(self):
        self.is_verifying = False
        self.last_result: Optional[Dict[str, Any]] = None
        self.latest_snapshot_bytes: Optional[bytes] = None
        self.latest_snapshot_status: str = "IDLE"  # IDLE, ANALYZING, VERIFIED, REVOKED
        self.last_latency_ms: float = 0.0

    def verify(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        candidate_gesture: str,
        candidate_count: int,
        other_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Dict[str, Any]:
        """Send snapshot to ChatGPT Vision for validation."""
        api_key = config.get("openai_api_key", "").strip()
        model = config.get("openai_model", "gpt-4o-mini").strip()

        if not api_key:
            logger.warning("No OpenAI API key configured, bypassing AI verification")
            return {
                "is_valid_gesture": True,
                "is_false_trigger": False,
                "gesture": candidate_gesture,
                "verified_finger_count": candidate_count,
                "confidence": 1.0,
                "rejection_reason": "None (No API Key)",
                "explanation": "AI verification skipped - no API key"
            }

        if self.is_verifying:
            logger.warning("ChatGPT verification already running. Rejecting overlapping request to maintain sequential delay.")
            return {
                "is_valid_gesture": False,
                "is_false_trigger": True,
                "rejection_reason": "Already Verifying",
                "explanation": "Concurrent verification rejected to guarantee proper sequential delay."
            }

        self.is_verifying = True
        self.latest_snapshot_status = "ANALYZING"
        start_time = time.time()

        try:
            # Create dual-view composite (Scene Context with all hands + High-Res Zoomed Candidate Hand)
            composite_snapshot = make_dual_view(frame_bgr, bbox, other_bboxes=other_bboxes)

            _, buf = cv2.imencode(".jpg", composite_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.latest_snapshot_bytes = buf.tobytes()
            b64_img = base64.b64encode(self.latest_snapshot_bytes).decode("utf-8")

            prompt = CHATGPT_SYSTEM_PROMPT.replace("{gesture_name}", candidate_gesture).replace("{finger_count}", str(candidate_count))

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 150,
                "temperature": 0.0,
            }

            with httpx.Client(timeout=5.5) as client:
                resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            self.last_latency_ms = round((time.time() - start_time) * 1000.0, 1)

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                result["latency_ms"] = self.last_latency_ms

                person_facing = result.get("person_facing_camera", result.get("person_looking_at_camera", True))
                hand_intentional = result.get("hand_intentionally_showing_gesture", True)
                is_valid = (
                    result.get("is_valid_gesture", False)
                    and not result.get("is_false_trigger", False)
                    and person_facing
                    and hand_intentional
                )
                self.latest_snapshot_status = "VERIFIED" if is_valid else "REVOKED"
                self.last_result = result

                # Stamp composite snapshot with colored verdict banner
                stamp_color = (0, 255, 120) if is_valid else (0, 0, 255)
                v_count = result.get("verified_finger_count", candidate_count)
                g_name = result.get("gesture", candidate_gesture)
                stamp_text = f"VERIFIED: {g_name} ({v_count}F)" if is_valid else f"REVOKED: {result.get('rejection_reason', 'False Trigger')}"
                stamped = composite_snapshot.copy()
                cv2.rectangle(stamped, (0, 0), (stamped.shape[1], 36), (15, 23, 42), -1)
                cv2.putText(stamped, stamp_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stamp_color, 2)
                _, v_buf = cv2.imencode(".jpg", stamped, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.latest_snapshot_bytes = v_buf.tobytes()

                return result
            else:
                logger.error(f"OpenAI error {resp.status_code}: {resp.text[:100]}")
                self.latest_snapshot_status = "ERROR"
                return {
                    "is_valid_gesture": False,
                    "is_false_trigger": True,
                    "rejection_reason": f"API Error {resp.status_code}",
                    "explanation": "OpenAI API returned non-200"
                }

        except Exception as e:
            logger.error(f"ChatGPT verification exception: {e}")
            self.latest_snapshot_status = "ERROR"
            return {
                "is_valid_gesture": False,
                "is_false_trigger": True,
                "rejection_reason": "Verification Exception",
                "explanation": str(e)
            }
        finally:
            self.is_verifying = False


chatgpt_verifier = ChatGPTVerifier()


# ========== HOME ASSISTANT CLIENT (REST + WEBSOCKET) ==========
class HAClient:
    """Home Assistant client supporting both WebSocket and REST API."""

    def __init__(self, url: str, token: str, entities: Dict[str, str], cooldown_seconds: float):
        self.url = url.rstrip("/")
        self.token = token.strip()
        self.entities = entities  # entity_1 .. entity_5
        self.cooldown_seconds = cooldown_seconds

        self.ws = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        self.last_trigger_time = 0.0
        self.last_action = "System Ready"
        self.entity_states: Dict[str, str] = {f"entity_{i}": "off" for i in range(1, 6)}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_rest = self.url.startswith("http://") or self.url.startswith("https://")

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    async def connect(self) -> bool:
        """Connect to Home Assistant via WebSocket or verify REST API."""
        if not self.token:
            logger.warning("No Home Assistant token configured")
            self.connected = False
            return False

        if self.is_rest:
            # Test REST API connection
            try:
                api_url = f"{self.url}/api/"
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(api_url, headers=self.get_headers(), timeout=4.0) as resp:
                        if resp.status == 200:
                            self.connected = True
                            logger.info(f"Connected to Home Assistant REST API at {self.url}")
                            await self.sync_states()
                            return True
                        else:
                            logger.warning(f"HA REST API returned HTTP {resp.status}")
                            self.connected = False
                            return False
            except Exception as e:
                logger.error(f"HA REST connection error: {e}")
                self.connected = False
                return False
        else:
            # WebSocket connection
            try:
                if self.session and not self.session.closed:
                    await self.session.close()

                self.session = aiohttp.ClientSession()
                self.ws = await self.session.ws_connect(self.url, timeout=5.0)

                msg = await self.ws.receive_json()
                if msg.get("type") != "auth_required":
                    self.connected = False
                    return False

                await self.ws.send_json({"type": "auth", "access_token": self.token})
                auth_resp = await self.ws.receive_json()

                if auth_resp.get("type") == "auth_ok":
                    self.connected = True
                    logger.info(f"Connected and authenticated to HA WebSocket at {self.url}")
                    await self.sync_states()
                    return True
                else:
                    self.connected = False
                    return False
            except Exception as e:
                logger.error(f"HA WebSocket connection error: {e}")
                self.connected = False
                return False

    async def sync_states(self):
        """Fetch current states for mapped entities."""
        if not self.connected:
            return

        try:
            base_http = self.url.replace("ws://", "http://").replace("wss://", "https://").replace("/api/websocket", "")
            api_url = f"{base_http}/api/states"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(api_url, headers=self.get_headers(), timeout=4.0) as resp:
                    if resp.status == 200:
                        all_states = await resp.json()
                        target_map = {v: k for k, v in self.entities.items() if v}
                        for item in all_states:
                            eid = item.get("entity_id")
                            if eid in target_map:
                                self.entity_states[target_map[eid]] = item.get("state", "off")
        except Exception as e:
            logger.debug(f"State sync error: {e}")

    async def toggle_entity(self, entity_num: int, entity_id: str) -> bool:
        """Toggle smart entity with debounce."""
        if not entity_id:
            return False

        now = time.time()
        if (now - self.last_trigger_time) < self.cooldown_seconds:
            logger.info(f"Cooldown active, skipping toggle for {entity_id}")
            return False

        base_http = self.url.replace("ws://", "http://").replace("wss://", "https://").replace("/api/websocket", "")
        api_url = f"{base_http}/api/services/light/toggle"

        domain = entity_id.split(".")[0]
        if domain != "light":
            api_url = f"{base_http}/api/services/{domain}/toggle"

        payload = {"entity_id": entity_id}

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(api_url, headers=self.get_headers(), json=payload, timeout=4.0) as resp:
                    if resp.status in [200, 201]:
                        self.last_trigger_time = now
                        key = f"entity_{entity_num}"
                        curr = self.entity_states.get(key, "off")
                        new_state = "off" if curr == "on" else "on"
                        self.entity_states[key] = new_state
                        self.last_action = f"Toggled {entity_id} -> {new_state.upper()}"
                        logger.info(f"⚡ Success: {self.last_action}")
                        return True
                    else:
                        logger.error(f"Failed to toggle {entity_id}: HTTP {resp.status}")
                        return False
        except Exception as e:
            logger.error(f"Exception toggling {entity_id}: {e}")
            return False

    async def reset_all(self) -> bool:
        """Reset / Turn OFF all configured devices."""
        base_http = self.url.replace("ws://", "http://").replace("wss://", "https://").replace("/api/websocket", "")
        success_count = 0

        for i in range(1, 6):
            eid = self.entities.get(f"entity_{i}")
            if not eid:
                continue
            domain = eid.split(".")[0]
            api_url = f"{base_http}/api/services/{domain}/turn_off"
            payload = {"entity_id": eid}
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(api_url, headers=self.get_headers(), json=payload, timeout=3.0) as resp:
                        if resp.status in [200, 201]:
                            self.entity_states[f"entity_{i}"] = "off"
                            success_count += 1
            except Exception:
                pass

        self.last_action = f"Reset All: Turned off {success_count} devices"
        logger.info(f"🔄 {self.last_action}")
        return True


# ========== REAL-TIME THREADED CAMERA STREAM READER ==========
class CameraStreamReader:
    """Reads frames from camera/RTSP in a dedicated thread to completely eliminate buffer delay and lag."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.last_frame_time = 0.0
        self.first_frame_logged = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open(self) -> bool:
        rtsp = config.get("rtsp_url", "").strip()
        cam_src = config.get("camera_source", "0").strip()

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if rtsp:
            # Handle unescaped '@' inside RTSP password (e.g. rtsp://admin:Admin@1234@192.168.2.41:554/...)
            if rtsp.startswith(("rtsp://", "rtsps://")):
                scheme, rest = rtsp.split("://", 1)
                slash_idx = rest.find("/")
                auth_host = rest[:slash_idx] if slash_idx != -1 else rest
                path_part = rest[slash_idx:] if slash_idx != -1 else ""
                if auth_host.count("@") > 1:
                    userinfo, hostpart = auth_host.rsplit("@", 1)
                    userinfo = userinfo.replace("@", "%40")
                    rtsp = f"{scheme}://{userinfo}@{hostpart}{path_part}"

            logger.info("Connecting to RTSP Camera stream...")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            self.cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            try:
                src_idx = int(cam_src)
            except ValueError:
                src_idx = 0
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(src_idx)

        if self.cap and self.cap.isOpened():
            if not rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("frame_width", 640))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("frame_height", 480))
                self.cap.set(cv2.CAP_PROP_FPS, config.get("fps", 20))
            self.is_connected = True
            self.last_frame_time = time.time()
            self.first_frame_logged = False
            logger.info("✅ Camera stream connected successfully")
            return True

        self.is_connected = False
        logger.error("❌ Failed to open camera device!")
        return False

    def _capture_loop(self):
        if not self._open():
            time.sleep(2)

        target_w = int(config.get("frame_width", 640))
        target_h = int(config.get("frame_height", 480))

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._open()
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                fh, fw = frame.shape[:2]
                if fw != target_w or fh != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if not self.first_frame_logged:
                    logger.info(f"📹 First camera frame received ({fw}x{fh} -> {target_w}x{target_h})")
                    self.first_frame_logged = True
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                    self.is_connected = True
            else:
                if time.time() - self.last_frame_time > 5.0:
                    logger.warning("Camera stream stalled for > 5s, reconnecting...")
                    self._open()
                    self.last_frame_time = time.time()
                time.sleep(0.01)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ========== MEDIAPIPE GESTURE & SKELETON PROCESSOR ==========
class GestureProcessor:
    """Zero-latency OpenCV & MediaPipe hand detector with ChatGPT AI verification bridge."""

    def __init__(self, ha_client: HAClient, sio_server: socketio.AsyncServer):
        self.ha_client = ha_client
        self.sio = sio_server
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera_reader = CameraStreamReader()

        try:
            self.mp_hands = mp.solutions.hands
        except AttributeError:
            try:
                import mediapipe.python.solutions.hands as mp_hands
                self.mp_hands = mp_hands
            except Exception as e:
                logger.error(f"Failed to load mediapipe hands: {e}")
                self.mp_hands = None
        self.hands = None

        # Stability buffer
        self.stability_target = int(config.get("stability_frames", 6))
        self.history = deque(maxlen=self.stability_target)
        self.current_stable_gesture: Optional[int] = None
        self.hand_detected = False
        self.stability_ratio = 0.0

        # Strict capture rate limiting & non-overlapping execution
        self.is_verifying_in_progress = False
        self.last_capture_time: float = 0.0
        self.last_completion_time: float = 0.0
        self.capture_delay_seconds: float = float(config.get("capture_delay_seconds", 1.0))
        self.cooldown_remaining: float = 0.0

        # State Machine: "READY" or "WAITING_FOR_RESET" (Require Fist or Rest before next gesture)
        self.pipeline_state = "READY"
        self.no_hand_frame_counter = 0
        self.no_gesture_frame_counter = 0
        self.fist_frame_counter = 0
        self.last_triggered_gesture: Optional[str] = None

        # Video stream frame holder (zero-copy pre-encoded for smooth streaming)
        self.latest_display_frame: Optional[np.ndarray] = None
        self.latest_jpeg_bytes: Optional[bytes] = self._make_status_jpeg("Connecting to Camera Stream...")
        self.frame_id: int = 1
        self.frame_lock = threading.Lock()

        # Stats
        self.current_fps = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @staticmethod
    def _make_status_jpeg(message: str) -> bytes:
        canvas = np.full((480, 640, 3), (30, 23, 15), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (640, 48), (15, 23, 42), -1)
        cv2.putText(canvas, "DRASIS GESTURE AI", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, message, (140, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (226, 232, 240), 2, cv2.LINE_AA)
        _, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def init_mediapipe(self):
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40,
            model_complexity=1
        )
        logger.info("MediaPipe Hands initialized with multi-hand 3D model (max 4 hands, complexity 1)")

    @staticmethod
    def _dist(p1: Any, p2: Any) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def count_fingers(self, landmarks: List[Any], w: int, h: int) -> Tuple[int, List[bool], Tuple[int, int, int, int]]:
        """Accurate evaluation of extended fingers:
        Fingertips must be UP-FACING or DIAGONAL UP-FACING (not side-facing, not downward-facing).
        Accurately recognizes 1..5 fingers including folded-thumb 4 fingers and all 3-finger styles.
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Bounding box
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        pad = 25
        bbox = (max(0, min(xs) - pad), max(0, min(ys) - pad), min(w, max(xs) + pad), min(h, max(ys) + pad))

        # Palm orientation check: Hand must be upright or diagonal-up
        # If palm is horizontal (side-facing) or pointing downwards, reject immediately as Rest/0 fingers
        palm_dy = wrist.y - middle_mcp.y  # Knuckles must be above wrist in camera image (dy > 0)
        palm_dx = abs(middle_mcp.x - wrist.x)
        if palm_dy <= 0.03 or palm_dy < 0.40 * palm_dx:
            return 0, [False, False, False, False, False], bbox

        # Scale-invariant normalization using palm length (wrist to middle_mcp)
        palm_scale = max(0.01, self._dist(wrist, middle_mcp))

        # 4 Main fingers: (tip, pip, mcp)
        # Index: (8, 6, 5), Middle: (12, 10, 9), Ring: (16, 14, 13), Pinky: (20, 18, 17)
        fingers_idx = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
        four_extended = []

        for tip_idx, pip_idx, mcp_idx in fingers_idx:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            d_wrist_tip = self._dist(wrist, tip)
            d_wrist_pip = self._dist(wrist, pip)
            d_mcp_tip = self._dist(mcp, tip)
            d_mcp_pip = self._dist(mcp, pip)

            # 1. Straightness: finger uncurled
            is_straight = (d_wrist_tip > d_wrist_pip * 1.02) and (d_mcp_tip > d_mcp_pip * 1.02)

            # 2. UP / DIAGONAL-UP ORIENTATION (NOT HORIZONTAL, NOT SIDE-FACING, NOT DOWNWARD):
            dy_up = mcp.y - tip.y
            dx_abs = abs(tip.x - mcp.x)

            # Tip must be above knuckle and pointing upright/diagonal-up (angle >= 24 degrees, dy_up >= 0.45 * dx_abs)
            # Purely horizontal or side-pointing fingers are rejected
            is_up_or_diagonal = (
                (tip.y <= pip.y + 0.04)
                and (dy_up > 0)
                and (dy_up >= 0.45 * dx_abs)
            )

            four_extended.append(is_straight and is_up_or_diagonal)

        index_ext, middle_ext, ring_ext, pinky_ext = four_extended
        num_four = sum(four_extended)

        # Thumb (tip=4, ip=3, mcp=2)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        index_mcp = landmarks[5]

        # Horizontal span of the knuckles (palm width)
        min_knuckle_x = min(index_mcp.x, pinky_mcp.x)
        max_knuckle_x = max(index_mcp.x, pinky_mcp.x)
        knuckle_span = max(0.01, max_knuckle_x - min_knuckle_x)

        # In an open 5-finger palm, thumb extends outward beyond the knuckle boundary.
        # When thumb is folded across the palm (for 4 fingers), thumb tip is inside the knuckle span!
        is_thumb_outside_palm = (
            thumb_tip.x < (min_knuckle_x - 0.05 * knuckle_span)
            or thumb_tip.x > (max_knuckle_x + 0.05 * knuckle_span)
        )

        d_thumb_pinky = self._dist(thumb_tip, pinky_mcp)
        d_ip_pinky = self._dist(thumb_ip, pinky_mcp)
        d_thumb_index = self._dist(thumb_tip, index_mcp)
        d_thumb_middle = self._dist(thumb_tip, middle_mcp)

        norm_thumb_index = d_thumb_index / palm_scale
        norm_thumb_middle = d_thumb_middle / palm_scale

        # Extended thumb: outside palm boundary, separated from knuckles, and farther from pinky than IP joint
        thumb_open = (
            is_thumb_outside_palm
            and (d_thumb_pinky > d_ip_pinky * 1.08)
            and (norm_thumb_index > 0.35)
            and (norm_thumb_middle > 0.32)
            and (thumb_tip.y < wrist.y)
        )

        extended = [thumb_open, index_ext, middle_ext, ring_ext, pinky_ext]

        # Natural Gesture Mapping
        if num_four == 0:
            count = 0  # Fist / Resting
        elif num_four == 1:
            if index_ext and thumb_open and norm_thumb_index > 0.38:
                count = 2  # L-shape: Thumb + Index
            else:
                count = 1  # 1 Finger: Index alone upright (☝️)
        elif num_four == 2:
            if thumb_open and norm_thumb_index > 0.35:
                count = 3  # 3 Fingers: Thumb + 2 fingers
            else:
                count = 2  # 2 Fingers: Peace sign (✌️)
        elif num_four == 3:
            if thumb_open and norm_thumb_index > 0.35:
                count = 4  # 4 Fingers: Thumb + 3 fingers
            else:
                count = 3  # 3 Fingers: Index + Middle + Ring with thumb folded
        elif num_four == 4:
            # 4 fingers are upright!
            # If thumb is folded into palm -> 4 Fingers (🖖)!
            # If thumb is extended out -> 5 Fingers (✋)!
            count = 5 if thumb_open else 4
        else:
            count = sum(four_extended) + (1 if thumb_open else 0)

        return count, extended, bbox

    def get_gesture_display(self, count: int) -> Tuple[str, str]:
        mapping = {
            0: ("Fist", "✊"),
            1: ("1 Finger", "☝️"),
            2: ("2 Fingers", "✌️"),
            3: ("3 Fingers", "👌"),
            4: ("4 Fingers", "🖖"),
            5: ("5 Fingers", "✋"),
        }
        return mapping.get(count, (f"{count} Fingers", "🖐️"))

    def start(self):
        self.running = True
        self.camera_reader.start()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("GestureProcessor thread started")

    def stop(self):
        self.running = False
        self.camera_reader.stop()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("GestureProcessor thread stopped")

    def _run_loop(self):
        self.init_mediapipe()

        while self.running:
            ret, frame = self.camera_reader.get_frame()
            if not ret or frame is None:
                time.sleep(0.015)
                continue

            # FPS counter
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = now

            # Mirror horizontally for natural webcam interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            clean_frame = frame.copy()

            # Fast direct RGB single-pass detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            self.hand_detected = False
            detected_count = -1
            bbox = None
            other_bboxes = []

            now = time.time()
            if self.last_completion_time > 0.0:
                time_since_completion = now - self.last_completion_time
                if time_since_completion < self.capture_delay_seconds:
                    self.cooldown_remaining = max(0.0, self.capture_delay_seconds - time_since_completion)
                else:
                    self.cooldown_remaining = 0.0
            else:
                self.cooldown_remaining = 0.0

            if results.multi_hand_landmarks:
                self.hand_detected = True
                self.no_hand_frame_counter = 0

                hand_candidates = []
                fist_detected = False

                # 1. Analyze and evaluate EVERY hand detected on screen
                for lms_obj in results.multi_hand_landmarks:
                    lms = lms_obj.landmark
                    h_count, h_extended, h_bbox = self.count_fingers(lms, w, h)
                    if h_count == 0:
                        fist_detected = True

                    area = 0
                    if h_bbox:
                        area = (h_bbox[2] - h_bbox[0]) * (h_bbox[3] - h_bbox[1])

                    hand_candidates.append({
                        "count": h_count,
                        "extended": h_extended,
                        "bbox": h_bbox,
                        "area": area,
                    })

                # 2. Check for active command gestures across all detected hands
                active_hands = [h for h in hand_candidates if h["count"] > 0]
                if active_hands:
                    # Select the primary active gesture hand (largest foreground area closest to camera)
                    best_hand = max(active_hands, key=lambda h: h["area"])
                    self.no_gesture_frame_counter = 0
                else:
                    best_hand = max(hand_candidates, key=lambda h: h["area"])
                    self.no_gesture_frame_counter += 1

                detected_count = best_hand["count"]
                bbox = best_hand["bbox"]
                other_bboxes = [h["bbox"] for h in hand_candidates if h["bbox"] and h is not best_hand]

                # 3. Render tracking bounding boxes and labels for ALL hands on screen
                for h_data in hand_candidates:
                    h_bbox = h_data["bbox"]
                    if not h_bbox:
                        continue
                    x1, y1, x2, y2 = h_bbox
                    is_primary = (h_data is best_hand and detected_count > 0)

                    # Distinct styling: Primary active gesture hand is Vibrant Cyan/Gold, other hands are Amber/Slate
                    box_color = (0, 215, 255) if is_primary else (200, 160, 50)
                    line_thick = 2 if is_primary else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, line_thick)
                    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 120, y1), (15, 23, 42), -1)

                    if h_data["count"] > 0:
                        track_text = f"Track: {h_data['count']}F" + (" *" if is_primary else "")
                    elif h_data["count"] == 0 and fist_detected:
                        track_text = "Track: Fist"
                    else:
                        track_text = "Track: Rest"

                    cv2.putText(
                        frame,
                        track_text,
                        (x1 + 6, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        box_color,
                        1
                    )

                # 4. Fist frames across detected hands
                if fist_detected:
                    self.fist_frame_counter += 1
                else:
                    self.fist_frame_counter = 0

                # Check if FIST on ANY hand resets pipeline from WAITING_FOR_RESET to READY
                if self.pipeline_state == "WAITING_FOR_RESET" and self.fist_frame_counter >= 4:
                    self.pipeline_state = "READY"
                    self.last_triggered_gesture = None
                    self.history.clear()
                    self.current_stable_gesture = None
                    logger.info("✊ Pipeline RESET by FIST gesture on detected hand! System READY for next gesture.")
                    if self.sio and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self.sio.emit("pipeline_reset", {"reason": "FIST", "state": "READY"}),
                            self.loop
                        )
                elif self.pipeline_state == "WAITING_FOR_RESET" and self.no_gesture_frame_counter >= 6:
                    self.pipeline_state = "READY"
                    self.last_triggered_gesture = None
                    self.history.clear()
                    self.current_stable_gesture = None
                    logger.info("🖐️ Pipeline RESET: Gesture hand lowered/rested! System READY for next gesture.")
                    if self.sio and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self.sio.emit("pipeline_reset", {"reason": "REST", "state": "READY"}),
                            self.loop
                        )

                # Accumulate stability history ONLY when pipeline is READY
                if (
                    self.pipeline_state == "READY"
                    and not self.is_verifying_in_progress
                    and not chatgpt_verifier.is_verifying
                    and self.cooldown_remaining == 0.0
                ):
                    self.history.append(detected_count)
                else:
                    self.history.clear()
                    self.stability_ratio = 0.0
            else:
                self.hand_detected = False
                self.history.clear()
                self.current_stable_gesture = None
                self.stability_ratio = 0.0
                self.fist_frame_counter = 0
                self.no_gesture_frame_counter += 1
                self.no_hand_frame_counter += 1

                # Check if REST (hand lowered / no hand for 5 frames ~0.25s) resets pipeline to READY
                if self.pipeline_state == "WAITING_FOR_RESET" and self.no_hand_frame_counter >= 5:
                    self.pipeline_state = "READY"
                    self.last_triggered_gesture = None
                    self.history.clear()
                    self.current_stable_gesture = None
                    logger.info("🖐️ Pipeline RESET by resting hand (hand lowered/removed)! System READY for next gesture.")
                    if self.sio and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self.sio.emit("pipeline_reset", {"reason": "REST", "state": "READY"}),
                            self.loop
                        )

            # Stability Analysis & ChatGPT Verification Trigger (Only in READY state)
            if (
                self.pipeline_state == "READY"
                and not self.is_verifying_in_progress
                and not chatgpt_verifier.is_verifying
                and self.cooldown_remaining == 0.0
            ):
                if len(self.history) >= self.stability_target:
                    most_common = Counter(self.history).most_common(1)[0]
                    stable_count = most_common[0]
                    ratio = most_common[1] / len(self.history)
                    self.stability_ratio = ratio

                    # Deliberate hold threshold (at least 80% frames must agree and finger > 0)
                    if ratio >= 0.80 and stable_count > 0:
                        gesture_name, gesture_icon = self.get_gesture_display(stable_count)
                        logger.info(f"📸 Candidate Gesture Stabilized: {gesture_name} ({stable_count} fingers). Capturing snapshot for ChatGPT AI...")

                        # ATOMIC LOCK: Lock immediately in main thread before launching verification
                        self.is_verifying_in_progress = True
                        self.last_capture_time = now
                        self.history.clear()

                        # Trigger verification in background thread to never block 30 FPS video
                        threading.Thread(
                            target=self._verify_and_execute,
                            args=(clean_frame, bbox, gesture_name, stable_count, other_bboxes),
                            daemon=True
                        ).start()
                else:
                    self.stability_ratio = len(self.history) / float(self.stability_target)
            else:
                self.stability_ratio = 0.0

            # Render on-screen HUD (Status, Verification Badge, FPS, Reset Prompt)
            self._render_hud(frame, detected_count)

            # Pre-encode JPEG once for instant zero-lag streaming
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 68])
            jpeg_bytes = jpeg.tobytes()

            with self.frame_lock:
                self.latest_display_frame = frame
                self.latest_jpeg_bytes = jpeg_bytes
                self.frame_id += 1

            time.sleep(0.005)

    def _render_hud(self, frame: np.ndarray, detected_count: int):
        h, w = frame.shape[:2]

        # Top Bar Background
        cv2.rectangle(frame, (0, 0), (w, 48), (15, 23, 42), -1)

        # AI Verification & Reset State Badge
        ai_status = chatgpt_verifier.latest_snapshot_status
        if self.is_verifying_in_progress or ai_status == "ANALYZING":
            badge_color = (0, 215, 255)
            badge_text = "AI: ANALYZING SNAPSHOT..."
        elif self.pipeline_state == "WAITING_FOR_RESET":
            badge_color = (0, 165, 255)
            badge_text = "RESET REQD: Make FIST [✊] or Lower Hand"
        elif self.cooldown_remaining > 0.0:
            badge_color = (0, 165, 255)
            badge_text = f"CAPTURE DELAY: {self.cooldown_remaining:.1f}s"
        elif ai_status == "VERIFIED":
            badge_color = (0, 255, 120)
            badge_text = "AI: READY (Show 1..5 Fingers)"
        elif ai_status == "REVOKED":
            badge_color = (0, 0, 255)
            reason = chatgpt_verifier.last_result.get("rejection_reason", "False Trigger") if chatgpt_verifier.last_result else "False Trigger"
            badge_text = f"AI: REVOKED ({reason[:22]})"
        else:
            badge_color = (148, 163, 184)
            badge_text = "AI: READY (Show 1..5 Fingers)"

        cv2.putText(frame, badge_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

        # Detected gesture badge
        if self.hand_detected and detected_count >= 0:
            if detected_count > 0:
                name, icon = self.get_gesture_display(detected_count)
                hud_text = f"Gesture: {name}"
                cv2.putText(frame, hud_text, (w - 230, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif self.fist_frame_counter >= 3:
                cv2.putText(frame, "Gesture: Fist [Reset]", (w - 260, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (148, 163, 184), 2)

        # Top progress bar (Hold progress or Cooldown)
        if self.stability_ratio > 0.0 and self.cooldown_remaining == 0.0 and not self.is_verifying_in_progress and self.pipeline_state == "READY":
            bar_w = int(w * self.stability_ratio)
            cv2.rectangle(frame, (0, 48), (bar_w, 52), (0, 215, 255), -1)
        elif self.pipeline_state == "WAITING_FOR_RESET":
            # Amber bar indicating reset is required
            cv2.rectangle(frame, (0, 48), (w, 52), (0, 165, 255), -1)
        elif self.cooldown_remaining > 0.0:
            cd_ratio = self.cooldown_remaining / max(0.1, self.capture_delay_seconds)
            bar_w = int(w * cd_ratio)
            cv2.rectangle(frame, (0, 48), (bar_w, 52), (0, 165, 255), -1)

    def _verify_and_execute(
        self,
        frame_bgr: np.ndarray,
        bbox: Any,
        gesture_name: str,
        finger_count: int,
        other_bboxes: Optional[List[Any]] = None
    ):
        """Asynchronously consult ChatGPT Vision, then either Trigger HA or Revoke action."""
        try:
            if self.sio and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit("ai_status", {
                        "status": "ANALYZING",
                        "gesture": gesture_name,
                        "fingers": finger_count,
                        "timestamp": datetime.now().isoformat()
                    }),
                    self.loop
                )

            ai_res = chatgpt_verifier.verify(frame_bgr, bbox, gesture_name, finger_count, other_bboxes=other_bboxes)

            is_valid = ai_res.get("is_valid_gesture", False)
            is_false = ai_res.get("is_false_trigger", False)
            reason = ai_res.get("rejection_reason", "None")
            explanation = ai_res.get("explanation", "")
            latency = ai_res.get("latency_ms", 0.0)

            person_facing = ai_res.get("person_facing_camera", ai_res.get("person_looking_at_camera", True))
            hand_intentional = ai_res.get("hand_intentionally_showing_gesture", True)

            # ========================================================
            # DECISION: EXECUTE IF TRUE, DO NOT EXECUTE IF FALSE
            # ========================================================
            if is_valid and not is_false and person_facing and hand_intentional:
                # VERIFY FINGERS AS WELL WITH AI: Override candidate count if AI verified different finger count
                ai_count = ai_res.get("verified_finger_count")
                if ai_count is not None and isinstance(ai_count, int) and 0 <= ai_count <= 5:
                    if ai_count != finger_count:
                        logger.info(f"🔍 AI Finger Count Override: MediaPipe candidate was {finger_count} fingers, but ChatGPT Vision verified {ai_count} fingers!")
                        finger_count = ai_count
                        gesture_name, _ = self.get_gesture_display(finger_count)

                logger.info(f"✅ GESTURE VERIFIED BY CHATGPT: {gesture_name} ({finger_count} fingers) [Latency: {latency}ms]")

                # Execute Home Assistant Action
                if finger_count == config.get("reset_gesture", 0):
                    asyncio.run_coroutine_threadsafe(self.ha_client.reset_all(), self.loop)
                    action_text = "Reset all devices"
                else:
                    entity_key = f"entity_{finger_count}"
                    entity_id = self.ha_client.entities.get(entity_key, "")
                    if entity_id:
                        asyncio.run_coroutine_threadsafe(
                            self.ha_client.toggle_entity(finger_count, entity_id),
                            self.loop
                        )
                        action_text = f"Toggled {entity_id}"
                    else:
                        action_text = f"No entity configured for {gesture_name}"

                # LOCK IN WAITING_FOR_RESET STATE
                self.pipeline_state = "WAITING_FOR_RESET"
                self.last_triggered_gesture = gesture_name
                logger.info(f"🔒 Triggered {gesture_name}! Locked in WAITING_FOR_RESET until user shows FIST ✊ or RESTS hand.")

                if self.sio and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit("gesture_verified", {
                            "gesture": gesture_name,
                            "fingers": finger_count,
                            "action": action_text,
                            "latency_ms": latency,
                            "explanation": explanation,
                            "pipeline_state": "WAITING_FOR_RESET",
                            "entity_states": self.ha_client.entity_states,
                            "timestamp": datetime.now().isoformat()
                        }),
                        self.loop
                    )
            else:
                logger.warning(f"🚫 GESTURE REVOKED BY CHATGPT: False Trigger Prevented! Reason: {reason} | {explanation}")
                self.pipeline_state = "READY"

                if self.sio and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.sio.emit("gesture_revoked", {
                            "gesture": gesture_name,
                            "fingers": finger_count,
                            "rejection_reason": reason,
                            "explanation": explanation,
                            "latency_ms": latency,
                            "pipeline_state": "READY",
                            "timestamp": datetime.now().isoformat()
                        }),
                        self.loop
                    )
        except Exception as ex:
            logger.error(f"Error in _verify_and_execute: {ex}")
            self.pipeline_state = "READY"
        finally:
            self.last_completion_time = time.time()
            self.is_verifying_in_progress = False
            self.current_stable_gesture = None
            self.history.clear()
            logger.info(f"⏳ Capture cooldown active: {self.capture_delay_seconds:.1f}s delay")
            if self.sio and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit("ai_status", {
                        "status": "COOLDOWN",
                        "cooldown_seconds": self.capture_delay_seconds,
                        "pipeline_state": self.pipeline_state,
                        "timestamp": datetime.now().isoformat()
                    }),
                    self.loop
                )


# ========== HTTP & WEBSOCKET SERVER ==========
class GestureServer:
    """aiohttp web application serving index UI, MJPEG video, snapshots, and REST endpoints."""

    def __init__(self):
        self.app = aiohttp.web.Application(middlewares=[self._path_normalizer_middleware])
        self.sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="aiohttp")
        self.sio.attach(self.app)

        self.ha_client: Optional[HAClient] = None
        self.gesture_processor: Optional[GestureProcessor] = None

        self._setup_routes()
        self._setup_sockets()

    @aiohttp.web.middleware
    async def _path_normalizer_middleware(self, request, handler):
        try:
            response = await handler(request)
            if response.status != 404:
                return response
        except aiohttp.web.HTTPNotFound:
            pass
        return await self.catch_all_handler(request)

    def _setup_routes(self):
        self.app.router.add_get("/", self.serve_index)
        self.app.router.add_get("/video", self.video_stream)
        self.app.router.add_get("/ws/video", self.video_ws)
        self.app.router.add_get("/frame", self.get_latest_frame)
        self.app.router.add_get("/snapshot/latest", self.get_latest_snapshot)
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/config", self.get_config)
        self.app.router.add_post("/api/toggle", self.api_toggle)
        self.app.router.add_post("/api/settings", self.api_settings)
        self.app.router.add_static("/static", str(BASE_DIR / "web"))
        # Catch-all route for HA Ingress (e.g. GET ////, /api/hassio_ingress/<token>/..., or any custom /path)
        self.app.router.add_route("*", "/{tail:.*}", self.catch_all_handler)

    async def catch_all_handler(self, request):
        """Route any HA Ingress multiple-slash or subpath requests cleanly without 404."""
        clean_path = re.sub(r"/+", "/", request.path).rstrip("/")
        if "/socket.io" in clean_path:
            return await self.sio.handle_request(request)
        if clean_path.endswith("/ws/video"):
            return await self.video_ws(request)
        if clean_path.endswith("/frame"):
            return await self.get_latest_frame(request)
        if clean_path.endswith("/video"):
            return await self.video_stream(request)
        if clean_path.endswith("/snapshot/latest"):
            return await self.get_latest_snapshot(request)
        if clean_path.endswith("/health"):
            return await self.health_check(request)
        if clean_path.endswith("/config"):
            return await self.get_config(request)
        if clean_path.endswith("/api/toggle") and request.method == "POST":
            return await self.api_toggle(request)
        if clean_path.endswith("/api/settings") and request.method == "POST":
            return await self.api_settings(request)
        return await self.serve_index(request)

    def _setup_sockets(self):
        @self.sio.on("connect")
        async def handle_connect(sid, environ):
            logger.info(f"Frontend connected: {sid}")
            await self.sio.emit("status", {
                "haConnected": self.ha_client.connected if self.ha_client else False,
                "aiEnabled": config.get("ai_verification_enabled", True),
                "entityStates": self.ha_client.entity_states if self.ha_client else {}
            })

    async def serve_index(self, request):
        index_file = BASE_DIR / "web" / "index.html"
        return aiohttp.web.FileResponse(index_file)

    async def video_ws(self, request):
        """Binary WebSocket video stream — bypasses HA Ingress / Cloudflare MJPEG boundary stripping."""
        ws = aiohttp.web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)

        last_id = -1
        try:
            while not ws.closed:
                frame_bytes = None
                if self.gesture_processor:
                    with self.gesture_processor.frame_lock:
                        if self.gesture_processor.latest_jpeg_bytes is not None:
                            if self.gesture_processor.frame_id != last_id:
                                frame_bytes = self.gesture_processor.latest_jpeg_bytes
                                last_id = self.gesture_processor.frame_id

                if frame_bytes is not None:
                    await ws.send_bytes(frame_bytes)
                await asyncio.sleep(0.04)  # ~25 FPS smooth binary push
        except (ConnectionResetError, asyncio.CancelledError, Exception):
            pass
        return ws

    async def get_latest_frame(self, request):
        """Return latest camera JPEG frame for universal HTTP fallback."""
        frame_bytes = None
        if self.gesture_processor:
            with self.gesture_processor.frame_lock:
                frame_bytes = self.gesture_processor.latest_jpeg_bytes
        if not frame_bytes:
            frame_bytes = GestureProcessor._make_status_jpeg("Waiting for Camera Frame...")
        return aiohttp.web.Response(
            body=frame_bytes,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache"
            }
        )

    async def video_stream(self, request):
        """Zero-latency smooth MJPEG video stream (works natively in any browser)."""
        response = aiohttp.web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        await response.prepare(request)

        last_id = -1
        try:
            while True:
                frame_bytes = None
                if self.gesture_processor:
                    with self.gesture_processor.frame_lock:
                        if self.gesture_processor.latest_jpeg_bytes is not None:
                            if self.gesture_processor.frame_id != last_id:
                                frame_bytes = self.gesture_processor.latest_jpeg_bytes
                                last_id = self.gesture_processor.frame_id

                if frame_bytes is not None:
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                    )
                    await response.write(header + frame_bytes + b"\r\n")
                await asyncio.sleep(0.035)
        except (ConnectionResetError, asyncio.CancelledError, Exception):
            pass

        return response

    async def get_latest_snapshot(self, request):
        """Return the latest analyzed snapshot image with verification result overlay."""
        if chatgpt_verifier.latest_snapshot_bytes:
            return aiohttp.web.Response(
                body=chatgpt_verifier.latest_snapshot_bytes,
                content_type="image/jpeg"
            )
        # Return fallback light-themed placeholder image
        blank = np.full((120, 160, 3), (249, 245, 241), dtype=np.uint8)
        cv2.putText(blank, "No Snapshot", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (139, 116, 100), 1, cv2.LINE_AA)
        _, j = cv2.imencode(".jpg", blank)
        return aiohttp.web.Response(body=j.tobytes(), content_type="image/jpeg")

    async def health_check(self, request):
        return aiohttp.web.json_response({
            "status": "ok",
            "ha_connected": self.ha_client.connected if self.ha_client else False,
            "entity_states": self.ha_client.entity_states if self.ha_client else {},
            "ai_verification_enabled": config.get("ai_verification_enabled", True),
            "camera_running": self.gesture_processor.running if self.gesture_processor else False,
            "fps": self.gesture_processor.current_fps if self.gesture_processor else 0,
            "uptime": datetime.now().isoformat()
        })

    async def get_config(self, request):
        return aiohttp.web.json_response({
            "entities": {
                "entity_1": config.get("entity_1", ""),
                "entity_2": config.get("entity_2", ""),
                "entity_3": config.get("entity_3", ""),
                "entity_4": config.get("entity_4", ""),
                "entity_5": config.get("entity_5", "")
            },
            "reset_gesture": config.get("reset_gesture", 0),
            "cooldown_seconds": config.get("cooldown_seconds", 1.5),
            "capture_delay_seconds": config.get("capture_delay_seconds", 1.0),
            "stability_frames": config.get("stability_frames", 6),
            "camera_source": config.get("camera_source", "0"),
            "rtsp_url": config.get("rtsp_url", ""),
            "openai_model": config.get("openai_model", "gpt-4o-mini"),
            "ai_verification_enabled": config.get("ai_verification_enabled", True)
        })

    async def api_toggle(self, request):
        data = await request.json()
        num = data.get("entity_num", 1)
        eid = config.get(f"entity_{num}", "")
        if self.ha_client and eid:
            success = await self.ha_client.toggle_entity(num, eid)
            return aiohttp.web.json_response({
                "success": success,
                "entity_states": self.ha_client.entity_states
            })
        return aiohttp.web.json_response({"success": False, "error": "Invalid entity"})

    async def api_settings(self, request):
        data = await request.json()
        for k, v in data.items():
            if k in config:
                config[k] = v
        save_config_to_env(config)
        return aiohttp.web.json_response({"success": True})

    async def start(self, stop_event: Optional[asyncio.Event] = None):
        logger.info("=" * 60)
        logger.info("🖐️ Starting Drasis Gesture Control Add-on + ChatGPT AI Verification")
        logger.info(f"🤖 OpenAI Model: {config.get('openai_model')}")
        logger.info(f"💡 Entities: 1={config.get('entity_1')}, 2={config.get('entity_2')}, 3={config.get('entity_3')}, 4={config.get('entity_4')}, 5={config.get('entity_5')}")
        logger.info(f"🔌 HA URL: {config.get('ha_url')}")
        logger.info("=" * 60)

        # Initialize HA Client
        entities = {f"entity_{i}": config.get(f"entity_{i}", "") for i in range(1, 6)}
        self.ha_client = HAClient(
            config.get("ha_url", ""),
            config.get("ha_token", ""),
            entities,
            config.get("cooldown_seconds", 1.5)
        )
        self.ha_client.set_loop(asyncio.get_event_loop())
        await self.ha_client.connect()

        # Initialize Gesture Processor
        self.gesture_processor = GestureProcessor(self.ha_client, self.sio)
        self.gesture_processor.set_loop(asyncio.get_event_loop())
        self.gesture_processor.start()

        # HA Keepalive loop
        asyncio.create_task(self._ha_keepalive())

        # Start aiohttp web server on port 8099
        runner = aiohttp.web.AppRunner(self.app)
        await runner.setup()
        port = config.get("port", 8099)
        site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
        await site.start()

        logger.info(f"🌐 Web Server started on http://localhost:{port}/")
        logger.info("=" * 60)

        if stop_event:
            await stop_event.wait()
        else:
            await asyncio.Event().wait()

    async def _ha_keepalive(self):
        while True:
            await asyncio.sleep(4.0)
            if self.ha_client:
                if not self.ha_client.connected:
                    await self.ha_client.connect()
                else:
                    await self.ha_client.sync_states()
                    await self.sio.emit("entity_states", {"states": self.ha_client.entity_states})


def main():
    server = GestureServer()
    stop_event = asyncio.Event()

    def _sig_handler(*_):
        logger.info("Received termination signal, shutting down...")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass

    try:
        asyncio.run(server.start(stop_event))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
    finally:
        if server.gesture_processor:
            server.gesture_processor.stop()


if __name__ == "__main__":
    main()
