# Real-Time Rasengan & Chidori Tracker 🌀⚡

An interactive computer vision project that casts **Rasengan** and **Chidori** onto your hands in real-time using **MediaPipe Hands**, **OpenCV**, and **Pygame**.

---

## Features

- **Dual-Hand Tracking:** Automatically binds **Rasengan** to the first hand raised and **Chidori** to the second hand.
- **Target Locking:** Tracks palm positions across frames to prevent effects from swapping when moving hands in and out of frame.
- **Screen Blending VFX:** Uses mathematical screen blending instead of naive additive blending to keep energy swirl details crisp without burning out to white.
- **Synced Audio:** Independent 2-channel audio playback via Pygame Mixer that starts and stops dynamically with hand gestures.

---

## Tech Stack

- **Python 3.9 - 3.12**
- **MediaPipe** (Hand landmark detection)
- **OpenCV** (Video frame processing & blending)
- **Pygame** (Multi-channel sound management)
- **NumPy**

---

## Project Structure

```text
Media-Pipe/
├── rasengan_video_overlay.py   # Main application logic
├── rasengan.mp4               # Rasengan VFX loop
├── chidori.mp4                # Chidori VFX loop
├── rasengan.mp3               # Audio SFX
├── chidori.mp3                # Audio SFX
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/kitalikedev/Media-Pipe.git](https://github.com/kitalikedev/Media-Pipe.git)
cd Media-Pipe
```

### 2. Create and activate a virtual environment
- **macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

- **Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python rasengan_video_overlay.py
```

---

## Controls

- **Open Hand:** Cast jutsu (Rasengan on 1st hand, Chidori on 2nd hand).
- **Close Hand / Lower Arm:** Cancel jutsu & reset playback.
- **`q` Key:** Quit application.
