"""
GestureFX Portal
-----------------
A real-time computer vision project that lets you control a "portal"
(a moving, resizable region with a visual filter applied inside it)
using nothing but your hand gestures — powered by OpenCV + MediaPipe.

Gestures:
    ☝  Index finger only        -> Move the portal (follows index fingertip of hand 1)
    🤏 Pinch (thumb+index close) -> Lock / unlock the portal position
    ☝ + ☝ (two hands, index up) -> Distance between both index fingertips -> resize portal
    ✌  Peace (index+middle up)  -> Cycle to the next filter
    ✊ Fist                      -> Turn the filter OFF (portal disappears)
    🖐 Open hand                 -> Turn the filter ON

Filters implemented:
    thermal, neon, night_vision, xray, cyberpunk, comic

Author: (you!) — built out of curiosity, not a product roadmap.
"""

import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720
MIN_PORTAL_R, MAX_PORTAL_R = 40, 300
DEFAULT_PORTAL_R = 120
PINCH_THRESHOLD = 0.045          # normalized distance (thumb tip <-> index tip)
GESTURE_COOLDOWN = 0.6           # seconds between peace-gesture filter switches
SMOOTHING = 6                    # frames of position smoothing for portal center

FILTER_NAMES = ["thermal", "neon", "night_vision", "xray", "cyberpunk", "comic"]


# --------------------------------------------------------------------------- #
# Filter functions — each takes a BGR frame region and returns a BGR frame
# --------------------------------------------------------------------------- #
def apply_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def apply_neon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    colored_edges = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    dark_bg = (img * 0.15).astype(np.uint8)
    mask = edges > 0
    out = dark_bg.copy()
    out[mask] = colored_edges[mask]
    return out


def apply_night_vision(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    noise = np.random.randint(0, 15, gray.shape, dtype=np.uint8)
    gray = cv2.add(gray, noise)
    green = np.zeros_like(img)
    green[:, :, 1] = gray
    return green


def apply_xray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    inv = cv2.GaussianBlur(inv, (3, 3), 0)
    return cv2.applyColorMap(inv, cv2.COLORMAP_BONE)


def apply_cyberpunk(img):
    b, g, r = cv2.split(img)
    r = cv2.add(r, 40)
    b = cv2.add(b, 60)
    g = cv2.subtract(g, 20)
    out = cv2.merge([b, g, r])
    glow = cv2.GaussianBlur(out, (0, 0), sigmaX=6)
    return cv2.addWeighted(out, 0.75, glow, 0.45, 0)


def apply_comic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


FILTER_FUNCS = {
    "thermal": apply_thermal,
    "neon": apply_neon,
    "night_vision": apply_night_vision,
    "xray": apply_xray,
    "cyberpunk": apply_cyberpunk,
    "comic": apply_comic,
}


# --------------------------------------------------------------------------- #
# Gesture helpers
# --------------------------------------------------------------------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_PIPS = [3, 6, 10, 14, 18]


def fingers_up(hand_landmarks, handedness_label):
    """Return a list of 5 booleans: [thumb, index, middle, ring, pinky] extended?"""
    lm = hand_landmarks.landmark
    up = []

    # Thumb: compare x, flipped depending on hand (mirror-friendly logic)
    if handedness_label == "Right":
        up.append(lm[4].x < lm[3].x)
    else:
        up.append(lm[4].x > lm[3].x)

    # Other four fingers: tip above pip (smaller y = higher on screen)
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        up.append(lm[tip].y < lm[pip].y)

    return up  # [thumb, index, middle, ring, pinky]


def classify_gesture(up):
    thumb, index, middle, ring, pinky = up
    if not any(up):
        return "fist"
    if all(up):
        return "open_hand"
    if index and middle and not ring and not pinky:
        return "peace"
    if index and not middle and not ring and not pinky:
        return "index_only"
    return "other"


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Portal state
    portal_center = [FRAME_W // 2, FRAME_H // 2]
    center_history = deque(maxlen=SMOOTHING)
    portal_radius = DEFAULT_PORTAL_R
    portal_locked = False
    filter_on = True
    filter_idx = 0
    last_gesture_switch_time = 0.0
    prev_pinch_state = False

    print("GestureFX Portal running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        index_tips_px = []          # for two-hand distance resizing
        primary_gesture = None
        pinch_active = False

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left"/"Right"
                up = fingers_up(hand_landmarks, label)
                gesture = classify_gesture(up)

                lm = hand_landmarks.landmark
                index_tip = lm[8]
                thumb_tip = lm[4]

                index_tips_px.append((int(index_tip.x * w), int(index_tip.y * h)))

                # Pinch detection (thumb tip close to index tip)
                if dist(thumb_tip, index_tip) < PINCH_THRESHOLD:
                    pinch_active = True

                # First detected hand drives move / open / fist / peace gestures
                if primary_gesture is None:
                    primary_gesture = gesture
                    if gesture == "index_only" and not portal_locked:
                        center_history.append((int(index_tip.x * w), int(index_tip.y * h)))

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Update portal center (smoothed) ---
        if center_history:
            avg_x = int(np.mean([p[0] for p in center_history]))
            avg_y = int(np.mean([p[1] for p in center_history]))
            portal_center = [avg_x, avg_y]

        # --- Pinch toggles lock (edge-triggered) ---
        if pinch_active and not prev_pinch_state:
            portal_locked = not portal_locked
        prev_pinch_state = pinch_active

        # --- Two-hand resize ---
        if len(index_tips_px) == 2:
            d = math.hypot(
                index_tips_px[0][0] - index_tips_px[1][0],
                index_tips_px[0][1] - index_tips_px[1][1],
            )
            portal_radius = int(np.clip(d / 2, MIN_PORTAL_R, MAX_PORTAL_R))

        # --- Peace switches filter (with cooldown) ---
        now = time.time()
        if primary_gesture == "peace" and (now - last_gesture_switch_time) > GESTURE_COOLDOWN:
            filter_idx = (filter_idx + 1) % len(FILTER_NAMES)
            last_gesture_switch_time = now

        # --- Fist / open hand toggle filter on/off ---
        if primary_gesture == "fist":
            filter_on = False
        elif primary_gesture == "open_hand":
            filter_on = True

        # --- Draw the portal ---
        output = frame.copy()
        if filter_on:
            cx, cy = portal_center
            r = portal_radius
            x1, y1 = max(cx - r, 0), max(cy - r, 0)
            x2, y2 = min(cx + r, w), min(cy + r, h)

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                filt_name = FILTER_NAMES[filter_idx]
                filtered_roi = FILTER_FUNCS[filt_name](roi)

                # Circular mask so the portal looks like a round window
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2),
                           min(roi.shape[0], roi.shape[1]) // 2, 255, -1)
                mask_3c = cv2.merge([mask, mask, mask])

                blended = np.where(mask_3c > 0, filtered_roi, roi)
                output[y1:y2, x1:x2] = blended

                ring_color = (0, 255, 255) if portal_locked else (255, 255, 255)
                cv2.circle(output, (cx, cy), r, ring_color, 3)
                cv2.putText(output, filt_name.upper(), (cx - r, max(cy - r - 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ring_color, 2)

        # --- HUD ---
        status = f"Filter: {'ON' if filter_on else 'OFF'} | Lock: {'YES' if portal_locked else 'no'} | Gesture: {primary_gesture or '-'}"
        cv2.putText(output, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(output, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(output, "q: quit", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("GestureFX Portal", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()