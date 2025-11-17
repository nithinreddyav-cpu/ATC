#!/usr/bin/env python3
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# =====================================================================
# MOTOR DRIVER (L298N)
# =====================================================================
LEFT_EN=12; LEFT_IN1=5; LEFT_IN2=6
RIGHT_EN=13; RIGHT_IN3=16; RIGHT_IN4=20

motor_pins = [LEFT_EN, LEFT_IN1, LEFT_IN2, RIGHT_EN, RIGHT_IN3, RIGHT_IN4]
for p in motor_pins:
    GPIO.setup(p, GPIO.OUT)

lpwm = GPIO.PWM(LEFT_EN, 1000)
rpwm = GPIO.PWM(RIGHT_EN, 1000)
lpwm.start(0)
rpwm.start(0)

def motors(left, right):
    # Left motor
    GPIO.output(LEFT_IN1, left >= 0)
    GPIO.output(LEFT_IN2, left < 0)
    # Right motor
    GPIO.output(RIGHT_IN3, right >= 0)
    GPIO.output(RIGHT_IN4, right < 0)

    lpwm.ChangeDutyCycle(abs(left))
    rpwm.ChangeDutyCycle(abs(right))

def forward(s=60): motors(s, s)
def backward(s=60): motors(-s, -s)
def left_turn(s=50): motors(-s, s)
def right_turn(s=50): motors(s, -s)
def stop(): motors(0, 0)

# =====================================================================
# ULTRASONIC SENSORS (3 for obstacles, 1 for pickup)
# =====================================================================
FRONT_TRIG=17; FRONT_ECHO=27
LEFT_TRIG=22; LEFT_ECHO=23
RIGHT_TRIG=24; RIGHT_ECHO=25
PICK_TRIG=8; PICK_ECHO=7

SENS = {
    "front": (FRONT_TRIG, FRONT_ECHO),
    "left":  (LEFT_TRIG, LEFT_ECHO),
    "right": (RIGHT_TRIG, RIGHT_ECHO),
    "pickup":(PICK_TRIG, PICK_ECHO)
}

for trig, echo in SENS.values():
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)

def dist(trig, echo):
    GPIO.output(trig, 0)
    time.sleep(0.0002)
    GPIO.output(trig, 1)
    time.sleep(0.00001)
    GPIO.output(trig, 0)

    start = time.time()
    end = time.time()

    timeout = start + 0.04
    while GPIO.input(echo) == 0 and time.time() < timeout:
        start = time.time()
    while GPIO.input(echo) == 1 and time.time() < timeout:
        end = time.time()

    return (end - start) * 17150

def read_sensors():
    return {
        "front": dist(FRONT_TRIG, FRONT_ECHO),
        "left":  dist(LEFT_TRIG, LEFT_ECHO),
        "right": dist(RIGHT_TRIG, RIGHT_ECHO),
        "pickup":dist(PICK_TRIG, PICK_ECHO)
    }

# =====================================================================
# SERVO (Trash pickup)
# =====================================================================
SERVO = 18
GPIO.setup(SERVO, GPIO.OUT)
servo = GPIO.PWM(SERVO, 50)
servo.start(0)

def set_servo(angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.4)
    servo.ChangeDutyCycle(0)

set_servo(90)  # default

# =====================================================================
# CAMERA SETUP
# =====================================================================
picam = Picamera2()
config = picam.create_video_configuration(main={"size": (320,240)})
picam.configure(config)
picam.start()
time.sleep(1)

bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

# =====================================================================
# CONSTANTS
# =====================================================================
OBSTACLE_LIMIT = 200   # cm (front, left, right)
PICKUP_LIMIT = 10      # cm (pickup sensor)
MIN_AREA = 500         # minimum contour area for trash

# =====================================================================
# MAIN LOOP
# =====================================================================
print("\n🚀 Autonomous Trash Bot Started...\n")

try:
    while True:

        # ---------------- CAMERA PROCESSING ----------------
        frame = picam.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)

        conts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        trash_list = []

        for c in conts:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = x + w//2

            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Plastic = white/light
            plastic = cv2.inRange(hsv_roi, (0,0,200), (180,40,255))
            # Metal = gray/reflective
            metal = cv2.inRange(hsv_roi, (0,0,80), (180,20,180))
            # Organic = green/brown → ignore
            organic = cv2.inRange(hsv_roi, (20,40,20), (80,255,150))

            px = cv2.countNonZero(plastic)
            mx = cv2.countNonZero(metal)
            ox = cv2.countNonZero(organic)

            if ox > px and ox > mx:
                continue  # ignore leaves, algae

            trash_list.append({"cx": cx, "area": area})

        # ----------------------------------------------------
        # READ ULTRASONIC SENSORS
        # ----------------------------------------------------
        d = read_sensors()
        front = d["front"]
        left  = d["left"]
        right = d["right"]
        pick  = d["pickup"]

        print(f"Front:{front:.1f}  Left:{left:.1f}  Right:{right:.1f} Pickup:{pick:.1f}")

        # ----------------------------------------------------
        # 1. PICKUP SENSOR → TRASH VERY CLOSE (<10 cm)
        # ----------------------------------------------------
        if pick < PICKUP_LIMIT:
            print("🟢 Trash at pickup sensor → Servo collecting...")
            stop()
            set_servo(180)
            time.sleep(1)
            set_servo(90)
            continue

        # ----------------------------------------------------
        # 2. OBSTACLE AVOIDANCE (3 sensors)
        # ----------------------------------------------------
        if front < OBSTACLE_LIMIT:
            print("⚠ Front obstacle → turning right")
            right_turn(50)
            time.sleep(0.5)
            continue

        if left < OBSTACLE_LIMIT:
            print("⚠ Left obstacle → turning right")
            right_turn(50)
            time.sleep(0.5)
            continue

        if right < OBSTACLE_LIMIT:
            print("⚠ Right obstacle → turning left")
            left_turn(50)
            time.sleep(0.5)
            continue

        # ----------------------------------------------------
        # 3. CAMERA FINDS TRASH
        # ----------------------------------------------------
        if len(trash_list) > 0:
            nearest = max(trash_list, key=lambda x: x["area"])
            cx = nearest["cx"]
            center = 320 // 2
            offset = cx - center

            if abs(offset) < 40:
                print("🎯 Trash centered → moving forward")
                forward(60)
            else:
                if offset < 0:
                    print("↪ Turning left toward trash")
                    left_turn(50)
                else:
                    print("↩ Turning right toward trash")
                    right_turn(50)

        else:
            # No trash → slow searching rotation
            print("🔍 Searching for trash...")
            left_turn(40)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n⚠ Stopping bot...")

finally:
    stop()
    servo.stop()
    GPIO.cleanup()
    picam.stop()

