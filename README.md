# GestureFX Portal 🚀

A real-time computer vision mini-project: control a moving, resizable
"portal" with visual filters — using only hand gestures. Built with
Python, OpenCV, and MediaPipe.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python gesturefx_portal.py
```

Needs a webcam. Press **q** to quit the app window.

## Gestures

| Gesture | Emoji | Action |
|---|---|---|
| Index finger only | ☝️ | Move the portal |
| Pinch (thumb + index touch) | 🤏 | Lock / unlock portal position |
| Two hands, index fingers up | ☝️+☝️ | Distance between fingertips controls portal size |
| Peace sign | ✌️ | Cycle to next filter |
| Fist | ✊ | Turn filter off |
| Open hand | 🖐️ | Turn filter on |

## Filters

`thermal` · `neon` · `night_vision` · `xray` · `cyberpunk` · `comic`

Cycle through them with the peace ✌️ gesture.

## How it works (short version)

1. **MediaPipe Hands** detects up to 2 hands per frame and gives 21 landmarks per hand.
2. A simple rule checks which fingertips are above their knuckle (`fingers_up`) to classify the gesture (fist, open hand, peace, index-only).
3. Pinch distance (thumb tip ↔ index tip) is measured in normalized coordinates to detect the lock/unlock pinch.
4. The portal is just a circular region of the frame where a chosen filter function replaces the pixels — the rest of the frame stays normal.
5. Position is smoothed over a few frames so the portal doesn't jitter.

## Known limitations / things to improve

- Gesture classification is rule-based (finger up/down), not a trained classifier — can misfire in odd hand angles.
- Only 2 hands supported (MediaPipe default), and lighting affects detection confidence.
- No gesture "hold to confirm" — a flickering pinch can toggle lock rapidly if held right at the pinch threshold.

Ideas to extend it: add a gesture history buffer to require holding a gesture for N frames before acting, add more filters, or save your favorite filter combo per gesture sequence.
