import time, math, threading
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0

SW, SH = pyautogui.size()
CAM_W, CAM_H = 640, 480

GAIN_X = 5.5
GAIN_Y = 7.5
JITTER_DEADBAND = 0.0006
ACCEL = 0.6
CLOSED_RATIO = 0.70
BURST_WINDOW = 0.60
MIN_BLINK, MAX_BLINK = 0.03, 0.45


class Cam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            r, f = self.cap.read()
            if r:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            f = self.frame
            self.frame = None
        return f

    def release(self):
        self.ok = False
        time.sleep(0.05)
        self.cap.release()


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=25.0, d_cutoff=1.0):
        self.mc, self.b, self.dc = min_cutoff, beta, d_cutoff
        self.x = None; self.dx = 0.0; self.t = None

    @staticmethod
    def _a(c, dt):
        tau = 1.0 / (2 * math.pi * c)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        d = (x - self.x) / dt
        ad = self._a(self.dc, dt)
        dh = ad * d + (1 - ad) * self.dx
        a = self._a(self.mc + self.b * abs(dh), dt)
        xh = a * x + (1 - a) * self.x
        self.x, self.dx, self.t = xh, dh, t
        return xh

    def reset(self):
        self.x = None; self.dx = 0.0; self.t = None


mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def ear(lm, idx):
    p = [np.array([lm[i].x * CAM_W, lm[i].y * CAM_H]) for i in idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return v / (2.0 * h + 1e-6)


def head_point(lm):
    ids = [1, 152, 10, 234, 454]
    return np.array([np.mean([lm[i].x for i in ids]),
                     np.mean([lm[i].y for i in ids])])


cam = Cam()
fxf, fyf = OneEuro(), OneEuro()

cx, cy = SW / 2, SH / 2
prev = None
ear_hist = deque(maxlen=150)
closed_f = open_f = 0
eye_closed = False
close_start = 0.0
blink_times = []
total = 0
action, action_t = "", 0.0

print("Running. Move head = move cursor, stop head = stop cursor.")
print("'c' = center cursor on screen | 'q' = quit | 4 blinks = exit")

while True:
    frame = cam.read()
    if frame is None:
        time.sleep(0.002)
        continue

    frame = cv2.flip(frame, 1)
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = time.time()
    quit_now = False
    cur_ear = 0.0

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        cur_ear = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2
        ear_hist.append(cur_ear)
        if len(ear_hist) >= 20:
            base = np.percentile(ear_hist, 90)
            if cur_ear < base * CLOSED_RATIO:
                closed_f += 1; open_f = 0
            else:
                open_f += 1; closed_f = 0

            if not eye_closed and closed_f >= 2:
                eye_closed = True
                close_start = t
            elif eye_closed and open_f >= 2:
                eye_closed = False
                if MIN_BLINK <= t - close_start <= MAX_BLINK:
                    total += 1
                    blink_times.append(t)
                    if len(blink_times) >= 4:
                        quit_now = True

        if blink_times and not eye_closed and t - blink_times[-1] > BURST_WINDOW:
            n = len(blink_times); blink_times.clear()
            if n == 2:
                pyautogui.click(); action, action_t = "CLICK", t
            elif n == 3:
                pyautogui.rightClick(); action, action_t = "RIGHT CLICK", t

        hp = head_point(lm)
        hx, hy = fxf(hp[0], t), fyf(hp[1], t)

        if prev is not None and not eye_closed:
            dx, dy = hx - prev[0], hy - prev[1]
            mag = math.hypot(dx, dy)
            if mag > JITTER_DEADBAND:
                boost = 1.0 + ACCEL * min(mag / 0.01, 3.0)
                cx += dx * GAIN_X * SW * boost
                cy += dy * GAIN_Y * SH * boost
                cx = float(np.clip(cx, 1, SW - 2))
                cy = float(np.clip(cy, 1, SH - 2))
                pyautogui.moveTo(cx, cy, _pause=False)
        prev = (hx, hy)

    cv2.putText(frame, f"Blinks: {total}  EAR: {cur_ear:.2f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if t - action_t < 1:
        cv2.putText(frame, action, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, "2=Click 3=Right 4=Exit | c=Center | q=Quit",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow("Head Control", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q") or quit_now:
        break
    if k == ord("c"):
        cx, cy = SW / 2, SH / 2
        pyautogui.moveTo(cx, cy, _pause=False)
        fxf.reset(); fyf.reset(); prev = None

cam.release()
cv2.destroyAllWindows()