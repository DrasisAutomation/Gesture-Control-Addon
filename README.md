# 🖐️ Gesture Control Add-on with ChatGPT AI Verification (v3.0)

Control smart home devices using hand gestures with **ChatGPT Vision AI (`gpt-4o-mini`) verification** to completely eliminate false triggers caused by:
- 📱 **Talking on a mobile phone / holding phone to ear**
- ☕ **Holding objects (coffee mug, cup, TV remote, pen, bottle)**
- 👤 **Touching face, head, chin, ear, or scratching**
- 🛋️ **Casual resting postures (chin in palm, arms folded)**
- 🚶 **Long-distance noise or ambiguous hands in the background**

Based on the [DrasisAutomation Gesture-Control-Addon](https://github.com/DrasisAutomation/Gesture-Control-Addon.git) architecture, upgraded with a dual-stage **MediaPipe ➔ ChatGPT AI Verification & Revocation Pipeline**.

---

## ⚡ How the False-Trigger Revocation Works

```
      [Camera Frame] (Webcam or RTSP)
             │
             ▼
      [MediaPipe 3D Hands]
             │ (Hand detected)
             ▼
   [Candidate Stability Check] (4 consecutive frames)
             │
             ▼
   [📸 Capture Context Snapshot]
             │
             ▼
 🤖 [Send Snapshot to ChatGPT Vision (gpt-4o-mini)]
             │
             ├── Is person holding a phone to their ear? ──> 🚫 REVOKE (Do not trigger)
             ├── Is person holding a mug / object? ────────> 🚫 REVOKE (Do not trigger)
             ├── Is person touching face / resting chin? ──> 🚫 REVOKE (Do not trigger)
             └── Is it a deliberate, clear hand sign? ─────> ✅ TRIGGER Home Assistant!
```

---

## 🎮 Gesture to Device Mapping

| Gesture | Finger Count | Default Entity | Action |
|---|---|---|---|
| **1 Finger** (Index up) | 1 | `light.pantry_light1` | Toggle Device 1 |
| **2 Fingers** (Peace) | 2 | `light.pantry_light2` | Toggle Device 2 |
| **3 Fingers** | 3 | `light.pantry_light3` | Toggle Device 3 |
| **4 Fingers** | 4 | `light.pantry_light4` | Toggle Device 4 |
| **5 Fingers** (Open Palm) | 5 | `light.pantry_light6` | Toggle Device 5 |
| **Fist** | 0 | All Devices | Reset (Turn OFF All) |

---

## 🚀 How to Run

### Option 1: Standalone / Windows PC (Instant Run)
Double-click `run_addon.bat` or run in terminal:

```bash
cd "c:\Camera Model\drasis_gesture_ai_addon"
..\backend\.venv\Scripts\python.exe app.py
```

Then open your browser at **`http://localhost:8099/`**.

### Option 2: Home Assistant OS Add-on
Copy this folder into your Home Assistant `/addons/local/` directory and install it directly from the Home Assistant Add-on Store.

---

## ⚙️ Configuration (`.env` or Add-on Options)

```env
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
AI_VERIFICATION_ENABLED=true

HA_URL=https://demo.lumihomepro1.com/
HA_TOKEN=eyJhbGciOi...

CAMERA_SOURCE=0
RTSP_URL=

ENTITY_1=light.pantry_light1
ENTITY_2=light.pantry_light2
ENTITY_3=light.pantry_light3
ENTITY_4=light.pantry_light4
ENTITY_5=light.pantry_light6
```
