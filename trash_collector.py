#!/usr/bin/env python3
"""
final_autonomous_trash_bot.py

Full autonomous loop:
- Continuous Picamera2 capture + immediate analysis (no 1s blocking)
- 3 ultrasonic sensors (front, left, right) for obstacle avoidance (200 cm)
- 1 ultrasonic pickup sensor for final close-range pickup (10 cm)
- Servo rotates 90 -> 180 -> 90 when pickup sensor triggers
- Save each captured frame to disk, and delete files older than 60 seconds
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
FRAME_FOLDER = "/home/saaketh/ATC/frames"   # ensure this path exists or will be created
FRAME_PREFIX = "frame_"                     # saved files: FRAME_PREFIX + timestamp.jpg
FRAME_RETENTION_SECONDS = 60                # auto-delete frames older than this
CAM_WIDTH, CAM_HEIGHT = 320, 240            # camera resolution for speed

# Ultrasonic distance thresholds (cm)
OBSTACLE_LIMIT = 200.0   # obstacle avoidance threshold (front/left/right)
PICKUP_LIMIT = 10.0      # pickup sensor threshold

# Detection parameters
MIN_AREA = 500           # minimum contour area considered as candidate trash
CENTER_TOLERANCE = 40    # pixels to consider "centered" horizontally
MOTOR_FORWARD_SPEED = 60
TURN_SPEED = 50

# GPIO pins (BCM) -- match your wiring
LEFT_EN = 12
LEFT_IN1 = 5
LEFT_IN2 = 6
RIGHT_EN = 13
RIGHT_IN3 = 16
RIGHT_IN4 = 20

# Ultrasonic (TRIG, ECHO)
FRONT_TRIG, FRONT_ECHO = 17, 27
LEFT_TRIG, LEFT_ECHO = 22, 23
RIGHT_TRIG, RIGHT_ECHO = 24, 25
PICK_TRIG, PICK_ECHO = 8, 7  # pickup sensor

# Servo for pickup (signal pin)
SERVO_PIN = 18

# Small sleep to yield CPU in main loop (<< 1s)
MAIN_LOOP_DELAY = 0.015  # 15 ms

# --------------------------- SETUP / UTIL ---------------------------
os.makedirs(FRAME_FOLDER, exist_ok=True)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins setup
motor_pins = [LEFT_EN, LEFT_IN1, LEFT_IN2, RIGHT_EN, RIGHT_IN3, RIGHT_IN4]
for p in motor_pins:
    GPIO.setup(p, GPIO.OUT)

left_pwm = GPIO.PWM(LEFT_EN, 1000)
right_pwm = GPIO.PWM(RIGHT_EN, 1000)
left_pwm.start(0)
right_pwm.start(0)

def motors_set(left_speed, right_speed):
    """Set motor direction and PWM. left_speed/right_speed in -100..100"""
    # left motor direction
    if left_speed >= 0:
        GPIO.output(LEFT_IN1, True)
        GPIO.output(LEFT_IN2, False)
    else:
        GPIO.output(LEFT_IN1, False)
        GPIO.output(LEFT_IN2, True)
    # right motor direction
    if right_speed >= 0:
        GPIO.output(RIGHT_IN3, True)
        GPIO.output(RIGHT_IN4, False)
    else:
        GPIO.output(RIGHT_IN3, False)
        GPIO.output(RIGHT_IN4, True)

    left_pwm.ChangeDutyCycle(min(100, max(0, abs(left_speed))))
    right_pwm.ChangeDutyCycle(min(100, max(0, abs(right_speed))))

def forward(speed=MOTOR_FORWARD_SPEED):
    motors_set(speed, speed)

def backward(speed=60):
    motors_set(-speed, -speed)

def turn_left(speed=TURN_SPEED):
    motors_set(-speed, speed)

def turn_right(speed=TURN_SPEED):
    motors_set(speed, -speed)

def stop():
    motors_set(0, 0)

# Ultrasonic setup
ultrasonic_sensors = {
    "front": (FRONT_TRIG, FRONT_ECHO),
    "left": (LEFT_TRIG, LEFT_ECHO),
    "right": (RIGHT_TRIG, RIGHT_ECHO),
    "pickup": (PICK_TRIG, PICK_ECHO)
}
for trig, echo in ultrasonic_sensors.values():
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)
    GPIO.output(trig, False)

def measure_distance(trig_pin, echo_pin, timeout_s=0.04):
    """Measure distance in cm using HC-SR04. Returns large number on timeout."""
    # Pulse trigger
    GPIO.output(trig_pin, False)
    time.sleep(0.0002)
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    start = time.time()
    timeout = start + timeout_s
    # wait for echo high
    while GPIO.input(echo_pin) == 0 and time.time() < timeout:
        start = time.time()
    if time.time() >= timeout:
        return 999.0
    # wait for echo low
    stop_t = time.time()
    timeout2 = time.time() + timeout_s
    while GPIO.input(echo_pin) == 1 and time.time() < timeout2:
        stop_t = time.time()
    if time.time() >= timeout2:
        return 999.0

    elapsed = stop_t - start
    distance_cm = elapsed * 17150.0
    return distance_cm

def read_all_distances():
    """Read all ultrasonic sensors. Returns dict of distances in cm (floats)."""
    d = {}
    # Read pickup first (fast), then other sensors
    for name, (trig, echo) in ultrasonic_sensors.items():
        try:
            d[name] = measure_distance(trig, echo)
        except Exception:
            d[name] = 999.0
    return d

# Servo setup
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)
def set_servo_angle(angle):
    """Angle in degrees 0..180"""
    duty = 2.0 + (angle / 18.0)
    servo.ChangeDutyCycle(duty)
    # short blocking to allow servo movement
    time.sleep(0.35)
    servo.ChangeDutyCycle(0)

# Camera setup (Picamera2)
picam = Picamera2()
camera_config = picam.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT)})
picam.configure(camera_config)
picam.start()
# allow sensor to warm
time.sleep(0.5)

bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

# Frame file management structures
frame_queue = collections.deque()  # store (timestamp, filepath)
frame_queue_lock = threading.Lock()
# Background thread that deletes files older than FRAME_RETENTION_SECONDS
def frame_cleanup_worker(stop_event):
    while not stop_event.is_set():
        now = time.time()
        removed = 0
        with frame_queue_lock:
            # Remove leftmost entries older than retention from both deque and disk
            while frame_queue and (now - frame_queue[0][0]) > FRAME_RETENTION_SECONDS:
                ts, fp = frame_queue.popleft()
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                        removed += 1
                except Exception:
                    pass
        # Sleep briefly; it's ok to wake once per 1s to cleanup, doesn't block main loop
        stop_event.wait(1.0)
    # final cleanup attempt when stopping
    with frame_queue_lock:
        while frame_queue:
            _, fp = frame_queue.popleft()
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass

# Start cleanup thread
cleanup_stop = threading.Event()
cleanup_thread = threading.Thread(target=frame_cleanup_worker, args=(cleanup_stop,), daemon=True)
cleanup_thread.start()

# --------------------------- Detection utilities ---------------------------
def classify_roi_type(roi_bgr):
    """
    Classify ROI as 'plastic', 'metal', 'organic', or 'unknown' using HSV heuristics.
    roi_bgr: numpy array (BGR)
    returns: str
    """
    try:
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return "unknown"
    # plastic: brighter, low saturation whites
    plastic_mask = cv2.inRange(hsv_roi, (0, 0, 200), (180, 50, 255))
    # metal: darker grayish reflective (low saturation, mid/value)
    metal_mask = cv2.inRange(hsv_roi, (0, 0, 80), (180, 40, 200))
    # organic: green / brown ranges
    organic_mask = cv2.inRange(hsv_roi, (10, 30, 20), (85, 255, 200))

    p_count = int(cv2.countNonZero(plastic_mask))
    m_count = int(cv2.countNonZero(metal_mask))
    o_count = int(cv2.countNonZero(organic_mask))

    # Decide
    if o_count > max(p_count, m_count) and o_count > 100:
        return "organic"
    if p_count > m_count and p_count > 150:
        return "plastic"
    if m_count > p_count and m_count > 120:
        return "metal"
    # fallback unknown
    return "unknown"

# --------------------------- Main loop ---------------------------

running = True
print("Autonomous trash bot starting (continuous capture, 60s frame retention)...")

try:
    # set default servo to 90 deg
    set_servo_angle(90)

    while running:
        loop_start = time.time()

        # 1) Capture frame immediately (no 1s delay)
        frame = None
        try:
            frame = picam.capture_array()
        except Exception as e:
            # camera capture transient error, skip this iteration but keep looping
            print("Camera capture error:", e)
            time.sleep(0.05)
            continue

        # 2) Save captured frame (compressed) to disk with timestamp
        ts = time.time()
        filename = f"{FRAME_PREFIX}{int(ts*1000)}.jpg"
        filepath = os.path.join(FRAME_FOLDER, filename)
        try:
            # encode to JPEG to reduce write size
            ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with open(filepath, 'wb') as f:
                    f.write(jpg.tobytes())
                with frame_queue_lock:
                    frame_queue.append((ts, filepath))
        except Exception:
            # if saving fails, continue analysis nevertheless
            pass

        # 3) Analyze frame: background subtraction + contour detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = bg_sub.apply(gray)
            fgmask = cv2.medianBlur(fgmask, 5)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            contours = []
            print("Frame processing error:", e)

        # Build list of candidate trash (plastic/metal) ignoring small / organic
        candidates = []
        for c in contours:
            try:
                area = cv2.contourArea(c)
                if area < MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # ensure ROI inside frame bounds
                x2, y2 = min(x+w, CAM_WIDTH-1), min(y+h, CAM_HEIGHT-1)
                roi = frame[y:y2, x:x2]
                if roi.size == 0:
                    continue
                obj_type = classify_roi_type(roi)
                if obj_type == "organic" or obj_type == "unknown":
                    continue
                cx = x + w//2
                candidates.append({"cx": cx, "area": area, "type": obj_type, "bbox": (x,y,w,h)})
            except Exception:
                continue

        # 4) Read ultrasonic sensors (pickup sensor + obstacles)
        distances = read_all_distances()  # returns dict with 'front','left','right','pickup'
        front_d = distances.get("front", 999.0)
        left_d = distances.get("left", 999.0)
        right_d = distances.get("right", 999.0)
        pickup_d = distances.get("pickup", 999.0)

        # 5) Check pickup sensor first (highest priority)
        if pickup_d < PICKUP_LIMIT:
            print(f"Pickup sensor triggered ({pickup_d:.1f} cm). Executing servo pickup.")
            stop()
            try:
                set_servo_angle(180)
                # small wait to ensure pickup
                time.sleep(0.6)
                set_servo_angle(90)
            except Exception as e:
                print("Servo error:", e)
            # after pickup continue loop immediately
            # do not sleep long; continue processing next frames
            continue

        # 6) Obstacle avoidance priority next
        obstacle_hit = False
        if front_d < OBSTACLE_LIMIT:
            print(f"Obstacle ahead at {front_d:.1f} cm -> turning right")
            turn_right(TURN_SPEED)
            time.sleep(0.35)
            stop()
            obstacle_hit = True
        elif left_d < OBSTACLE_LIMIT:
            print(f"Obstacle left at {left_d:.1f} cm -> turning right")
            turn_right(TURN_SPEED)
            time.sleep(0.35)
            stop()
            obstacle_hit = True
        elif right_d < OBSTACLE_LIMIT:
            print(f"Obstacle right at {right_d:.1f} cm -> turning left")
            turn_left(TURN_SPEED)
            time.sleep(0.35)
            stop()
            obstacle_hit = True

        if obstacle_hit:
            # continue immediately to next capture (no 1s delay)
            continue

        # 7) If we have camera-detected candidates, pick nearest (largest area)
        if candidates:
            target = max(candidates, key=lambda x: x["area"])
            cx = target["cx"]
            frame_center = CAM_WIDTH // 2
            offset = cx - frame_center

            # align: turn if needed, else move forward
            if abs(offset) <= CENTER_TOLERANCE:
                # move forward, but check front sensor frequently
                # keep motors on for a short time slice then re-evaluate
                print(f"Moving toward trash ({target['type']}) offset={offset}")
                forward(MOTOR_FORWARD_SPEED)
                time.sleep(0.08)  # short move slice, <100ms
                stop()
            else:
                if offset < 0:
                    print(f"Turning left to align (offset {offset})")
                    turn_left(TURN_SPEED)
                    time.sleep(0.06)
                    stop()
                else:
                    print(f"Turning right to align (offset {offset})")
                    turn_right(TURN_SPEED)
                    time.sleep(0.06)
                    stop()
        else:
            # No detected valid trash: do a slow scan rotation to search
            # short rotation then re-evaluate (non-blocking)
            turn_left(25)
            time.sleep(0.07)
            stop()

        # Yield a small amount (very short) so loop isn't 100% CPU
        elapsed = time.time() - loop_start
        if elapsed < MAIN_LOOP_DELAY:
            time.sleep(MAIN_LOOP_DELAY - elapsed)

except KeyboardInterrupt:
    print("KeyboardInterrupt - shutting down")

except Exception as e:
    print("Unhandled error in main loop:", repr(e))

finally:
    # shutdown / cleanup
    try:
        stop()
        servo.ChangeDutyCycle(0)
        servo.stop()
    except Exception:
        pass
    try:
        left_pwm.stop()
        right_pwm.stop()
    except Exception:
        pass

    # stop cleanup thread and clean remaining frames
    cleanup_stop.set()
    cleanup_thread.join(timeout=2.0)

    # remove any remaining frames older than 0s (attempt)
    with frame_queue_lock:
        while frame_queue:
            _, fp = frame_queue.popleft()
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass

    try:
        GPIO.cleanup()
    except Exception:
        pass

    try:
        picam.stop()
    except Exception:
        pass

    print("Shutdown complete.")

