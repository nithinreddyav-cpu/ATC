#!/usr/bin/env python3
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ------------------------------------------------------------
# GPIO SETUP
# ------------------------------------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# L298N Motor pins
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

# Servos
SERVO_DOOR = 18
SERVO_ARM = 19
GPIO.setup(SERVO_DOOR, GPIO.OUT)
GPIO.setup(SERVO_ARM, GPIO.OUT)

servo_door = GPIO.PWM(SERVO_DOOR, 50)
servo_arm = GPIO.PWM(SERVO_ARM, 50)
servo_door.start(0)
servo_arm.start(0)

# ------------------------------------------------------------
# MOTOR CONTROL FUNCTIONS
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
# ULTRASONIC SENSOR FUNCTION
# ------------------------------------------------------------
def measure_distance(trig, echo):
    GPIO.output(trig, False)
    time.sleep(0.0002)

    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    start = time.time()
    stop = time.time()

    timeout = time.time() + 0.04  # 40ms

    while GPIO.input(echo) == 0 and time.time() < timeout:
        start = time.time()

    while GPIO.input(echo) == 1 and time.time() < timeout:
        stop = time.time()

    elapsed = stop - start
    return elapsed * 17150  # Convert to cm

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
# CAMERA SETUP USING PICAMERA2
# ------------------------------------------------------------
picam = Picamera2()
config = picam.create_video_configuration(main={"size": (320, 240)})
picam.configure(config)
picam.start()
time.sleep(1)

bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)
MIN_AREA = 800

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    print("🚀 Autonomous Trash Collector (Picamera2) Started...")

    while True:
        img = picam.capture_array()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fg = bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            time.sleep(0.25)
            stop()
            continue

        # Nearest trash = biggest area
        target = max(detected, key=lambda x: x["area"])
        cx = target["cx"]

        # --------------- COLOR CLASSIFICATION ----------------
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
            print("Organic → ignored")
            continue

        print("Nearest trash:", trash_type)

        # ---------------- ALIGN AND MOVE ----------------
        frame_center = 320 // 2
        offset = cx - frame_center

        if abs(offset) < 40:
            print("Aligned → moving forward")
            forward(60)
        else:
            if offset < 0:
                print("Turning left")
                turn_left(45)
            else:
                print("Turning right")
                turn_right(45)

        distances = read_all_sensors()
        front = distances["front"]

        if front < 30:
            print("Close to trash → Collecting")
            stop()
            open_collector()
            time.sleep(1.5)
            close_collector()
            backward(60)
            time.sleep(1)
            stop()

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
    picam.stop()
