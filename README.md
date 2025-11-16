#!/usr/bin/env python3
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera import PiCamera
from picamera.array import PiRGBArray

# ------------------------------------------------------------
# GPIO SETUP
# ------------------------------------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# L298N Motor Driver Pins
LEFT_EN = 12
LEFT_IN1 = 5
LEFT_IN2 = 6
RIGHT_EN = 13
RIGHT_IN3 = 16
RIGHT_IN4 = 20

MOTOR_PINS = [LEFT_EN, LEFT_IN1, LEFT_IN2, RIGHT_EN, RIGHT_IN3, RIGHT_IN4]
for pin in MOTOR_PINS:
    GPIO.setup(pin, GPIO.OUT)

left_pwm = GPIO.PWM(LEFT_EN, 1000)
right_pwm = GPIO.PWM(RIGHT_EN, 1000)
left_pwm.start(0)
right_pwm.start(0)

# Ultrasonic sensors: front, left, right, back
SENSORS = {
    "front": (17, 27),
    "left": (22, 23),
    "right": (24, 25),
    "back": (8, 7)
}

for trig, echo in SENSORS.values():
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)

# Servos for collector
SERVO_DOOR = 18
SERVO_ARM = 19
GPIO.setup(SERVO_DOOR, GPIO.OUT)
GPIO.setup(SERVO_ARM, GPIO.OUT)

servo_door = GPIO.PWM(SERVO_DOOR, 50)
servo_arm = GPIO.PWM(SERVO_ARM, 50)
servo_door.start(0)
servo_arm.start(0)

# ------------------------------------------------------------
# MOTOR CONTROL
# ------------------------------------------------------------
def set_motor(left, right):
    if left >= 0:
        GPIO.output(LEFT_IN1, True)
        GPIO.output(LEFT_IN2, False)
    else:
        GPIO.output(LEFT_IN1, False)
        GPIO.output(LEFT_IN2, True)

    if right >= 0:
        GPIO.output(RIGHT_IN3, True)
        GPIO.output(RIGHT_IN4, False)
    else:
        GPIO.output(RIGHT_IN3, False)
        GPIO.output(RIGHT_IN4, True)

    left_pwm.ChangeDutyCycle(abs(left))
    right_pwm.ChangeDutyCycle(abs(right))

def forward(s=50): set_motor(s, s)
def backward(s=50): set_motor(-s, -s)
def turn_left(s=45): set_motor(-s, s)
def turn_right(s=45): set_motor(s, -s)
def stop(): set_motor(0, 0)

# ------------------------------------------------------------
# ULTRASONIC
# ------------------------------------------------------------
def measure_distance(trig, echo):
    GPIO.output(trig, False)
    time.sleep(0.0002)
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    start, end = 0, 0
    timeout = time.time() + 0.04

    while GPIO.input(echo) == 0 and time.time() < timeout:
        start = time.time()

    while GPIO.input(echo) == 1 and time.time() < timeout:
        end = time.time()

    duration = end - start
    return duration * 17150  # cm

def read_all_sensors():
    dist = {}
    for key, pins in SENSORS.items():
        dist[key] = measure_distance(pins[0], pins[1])
    return dist

# ------------------------------------------------------------
# SERVO CONTROL
# ------------------------------------------------------------
def set_servo(pwm, angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)

def open_collector():
    set_servo(servo_door, 90)
    set_servo(servo_arm, 45)

def close_collector():
    set_servo(servo_arm, 0)
    set_servo(servo_door, 0)

# ------------------------------------------------------------
# CAMERA SETUP
# ------------------------------------------------------------
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 16
raw = PiRGBArray(camera, size=(320, 240))
time.sleep(1)

bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
MIN_AREA = 800

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    print("🚀 Autonomous Trash Collector Started...")

    for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        fg = bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ------------------------------------------------------------
        # MULTI-OBJECT DETECTION → SELECT NEAREST
        # ------------------------------------------------------------
        detected = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

            detected.append({
                "cx": cx,
                "cy": cy,
                "area": area
            })

        if len(detected) == 0:
            print("Searching for trash...")
            turn_left(40)
            time.sleep(0.3)
            stop()
            raw.truncate(0)
            continue

        # Select nearest (largest area)
        target = max(detected, key=lambda x: x["area"])
        cx = target["cx"]

        # ------------------------------------------------------------
        # COLOR-BASED TRASH IDENTIFICATION
        # ------------------------------------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        plastic_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        metal_mask = cv2.inRange(hsv, (0, 0, 80), (180, 40, 200))
        organic_mask = cv2.inRange(hsv, (20, 20, 20), (40, 255, 200))

        plastic = cv2.countNonZero(plastic_mask)
        metal = cv2.countNonZero(metal_mask)
        organic = cv2.countNonZero(organic_mask)

        trash_type = "unknown"
        if plastic > 300:
            trash_type = "plastic"
        elif metal > 300:
            trash_type = "metal"
        elif organic > 300:
            trash_type = "organic"

        if trash_type == "organic":
            print("Organic detected → ignoring...")
            raw.truncate(0)
            continue

        print("Nearest trash detected:", trash_type)

        # ------------------------------------------------------------
        # ALIGN TO TRASH
        # ------------------------------------------------------------
        center = img.shape[1] // 2
        offset = cx - center

        if abs(offset) < 40:
            print("Aligned → moving forward")
            forward(55)
        else:
            if offset < 0:
                print("Turning left...")
                turn_left(45)
            else:
                print("Turning right...")
                turn_right(45)

        # ------------------------------------------------------------
        # OBSTACLE CHECK + COLLECTION
        # ------------------------------------------------------------
        dist = read_all_sensors()
        front = dist["front"]

        if front < 28:
            print("Close to trash → collecting")
            stop()
            open_collector()
            time.sleep(1.5)
            close_collector()
            backward(60)
            time.sleep(1)
            stop()

        raw.truncate(0)
        time.sleep(0.05)


try:
    main()
except KeyboardInterrupt:
    print("Stopping robot...")
finally:
    stop()
    left_pwm.stop()
    right_pwm.stop()
    servo_door.stop()
    servo_arm.stop()
    GPIO.cleanup()
