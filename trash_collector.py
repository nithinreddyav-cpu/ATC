#!/usr/bin/env python3
"""
final_autonomous_trash_bot.py

Full autonomous loop:
- Continuous Picamera2 capture + immediate analysis (no 1s blocking)
- 3 ultrasonic sensors (front, left, right) for obstacle avoidance (200 cm)
- 1 ultrasonic pickup sensor for final close-range pickup (10 cm)
- Servo rotates 90 -> 180 -> 90 when pickup sensor triggers
- Save each captured frame to disk, auto-delete older than 60 seconds
- Robust error handling and clean shutdown
"""

import os
import time
import cv2
import numpy as np
import threading
import collections
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ------------------------------ CONFIG ------------------------------
FRAME_FOLDER = "/home/saaketh/ATC/frames"
FRAME_PREFIX = "frame_"
FRAME_RETENTION_SECONDS = 60
CAM_WIDTH, CAM_HEIGHT = 320, 240

OBSTACLE_LIMIT = 200.0
PICKUP_LIMIT = 10.0

MIN_AREA = 500
CENTER_TOLERANCE = 40
MOTOR_FORWARD_SPEED = 60
TURN_SPEED = 50

# UPDATED SLEEP TIME → 25 milliseconds
MAIN_LOOP_DELAY = 0.025   # 25 ms CPU yield

# GPIO pin mapping (BCM)
LEFT_EN = 12
LEFT_IN1 = 5
LEFT_IN2 = 6
RIGHT_EN = 13
RIGHT_IN3 = 14
RIGHT_IN4 = 20

FRONT_TRIG, FRONT_ECHO = 17, 27
LEFT_TRIG, LEFT_ECHO = 22, 23
RIGHT_TRIG, RIGHT_ECHO = 24, 25
PICK_TRIG, PICK_ECHO = 8, 7

SERVO_PIN = 18

# --------------------------- SETUP / UTIL ---------------------------
os.makedirs(FRAME_FOLDER, exist_ok=True)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [LEFT_EN, LEFT_IN1, LEFT_IN2, RIGHT_EN, RIGHT_IN3, RIGHT_IN4]
for p in motor_pins:
    GPIO.setup(p, GPIO.OUT)

left_pwm = GPIO.PWM(LEFT_EN, 1000)
right_pwm = GPIO.PWM(RIGHT_EN, 1000)
left_pwm.start(0)
right_pwm.start(0)

def motors_set(left_speed, right_speed):
    if left_speed >= 0:
        GPIO.output(LEFT_IN1, True)
        GPIO.output(LEFT_IN2, False)
    else:
        GPIO.output(LEFT_IN1, False)
        GPIO.output(LEFT_IN2, True)

    if right_speed >= 0:
        GPIO.output(RIGHT_IN3, True)
        GPIO.output(RIGHT_IN4, False)
    else:
        GPIO.output(RIGHT_IN3, False)
        GPIO.output(RIGHT_IN4, True)

    left_pwm.ChangeDutyCycle(min(100, abs(left_speed)))
    right_pwm.ChangeDutyCycle(min(100, abs(right_speed)))

def forward(s=MOTOR_FORWARD_SPEED): motors_set(s, s)
def backward(s=60): motors_set(-s, -s)
def turn_left(s=TURN_SPEED): motors_set(-s, s)
def turn_right(s=TURN_SPEED): motors_set(s, -s)
def stop(): motors_set(0, 0)

ultrasonic_sensors = {
    "front": (FRONT_TRIG, FRONT_ECHO),
    "left":  (LEFT_TRIG, LEFT_ECHO),
    "right": (RIGHT_TRIG, RIGHT_ECHO),
    "pickup":(PICK_TRIG, PICK_ECHO)
}

for trig, echo in ultrasonic_sensors.values():
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)
    GPIO.output(trig, False)

def measure_distance(trig_pin, echo_pin, timeout_s=0.04):
    GPIO.output(trig_pin, False)
    time.sleep(0.0002)
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    start = time.time()
    timeout = start + timeout_s
    while GPIO.input(echo_pin) == 0 and time.time() < timeout:
        start = time.time()

    if time.time() >= timeout:
        return 999.0

    stop_t = time.time()
    timeout2 = time.time() + timeout_s
    while GPIO.input(echo_pin) == 1 and time.time() < timeout2:
        stop_t = time.time()

    if time.time() >= timeout2:
        return 999.0

    return (stop_t - start) * 17150.0

def read_all_distances():
    d = {}
    for name, (trig, echo) in ultrasonic_sensors.items():
        try:
            d[name] = measure_distance(trig, echo)
        except Exception:
            d[name] = 999.0
    return d

GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

def set_servo(angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.35)
    servo.ChangeDutyCycle(0)

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT)})
picam.configure(config)
picam.start()
time.sleep(0.5)

bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

frame_queue = collections.deque()
frame_queue_lock = threading.Lock()
cleanup_stop = threading.Event()

def cleanup_worker():
    while not cleanup_stop.is_set():
        now = time.time()
        with frame_queue_lock:
            while frame_queue and (now - frame_queue[0][0]) > FRAME_RETENTION_SECONDS:
                _, fp = frame_queue.popleft()
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except:
                    pass
        cleanup_stop.wait(1.0)

cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
cleanup_thread.start()

def classify_roi(roi):
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except:
        return "unknown"

    plastic = cv2.inRange(hsv, (0,0,200), (180,40,255))
    metal   = cv2.inRange(hsv, (0,0,80),  (180,40,200))
    organic = cv2.inRange(hsv, (10,30,20),(85,255,200))

    p = cv2.countNonZero(plastic)
    m = cv2.countNonZero(metal)
    o = cv2.countNonZero(organic)

    if o > max(p,m) and o > 100: return "organic"
    if p > m and p > 150: return "plastic"
    if m > p and m > 120: return "metal"
    return "unknown"

print("Autonomous trash bot running with 25ms CPU sleep...")

try:
    set_servo(90)

    while True:
        loop_start = time.time()

        # --- Capture frame ---
        try:
            frame = picam.capture_array()
        except:
            time.sleep(0.02)
            continue

        # --- Save frame ---
        ts = time.time()
        fp = os.path.join(FRAME_FOLDER, f"{FRAME_PREFIX}{int(ts*1000)}.jpg")
        try:
            ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with open(fp, 'wb') as f:
                    f.write(jpg.tobytes())
                with frame_queue_lock:
                    frame_queue.append((ts, fp))
        except:
            pass

        # --- Process frame ---
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bg_sub.apply(gray)
            fg = cv2.medianBlur(fg, 5)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours = []

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue

            typ = classify_roi(roi)
            if typ in ("plastic", "metal"):
                cx = x + w//2
                candidates.append((area, cx, typ))

        # --- Read ultrasonic sensors ---
        d = read_all_distances()
        front = d["front"]
        left  = d["left"]
        right = d["right"]
        pickup= d["pickup"]

        # --- Pickup action ---
        if pickup < PICKUP_LIMIT:
            stop()
            set_servo(180)
            time.sleep(0.6)
            set_servo(90)
            continue

        # --- Obstacle avoidance ---
        if front < OBSTACLE_LIMIT:
            turn_right(TURN_SPEED); time.sleep(0.3); stop(); continue
        if left < OBSTACLE_LIMIT:
            turn_right(TURN_SPEED); time.sleep(0.3); stop(); continue
        if right < OBSTACLE_LIMIT:
            turn_left(TURN_SPEED);  time.sleep(0.3); stop(); continue

        # --- Trash alignment and movement ---
        if candidates:
            # choose largest area (nearest)
            area, cx, typ = max(candidates)

            offset = cx - (CAM_WIDTH//2)
            if abs(offset) < CENTER_TOLERANCE:
                forward(MOTOR_FORWARD_SPEED)
                time.sleep(0.08)
                stop()
            else:
                if offset < 0:
                    turn_left(TURN_SPEED)
                    time.sleep(0.06)
                    stop()
                else:
                    turn_right(TURN_SPEED)
                    time.sleep(0.06)
                    stop()
        else:
            turn_left(25)
            time.sleep(0.07)
            stop()

        # --- 25 ms sleep (updated as you requested) ---
        elapsed = time.time() - loop_start
        if elapsed < MAIN_LOOP_DELAY:
            time.sleep(MAIN_LOOP_DELAY - elapsed)

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    stop()
    try:
        servo.stop()
        left_pwm.stop()
        right_pwm.stop()
    except:
        pass
    cleanup_stop.set()
    cleanup_thread.join(timeout=2)
    GPIO.cleanup()
    picam.stop()
    print("Shutdown complete.")
