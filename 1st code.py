#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
import RPi.GPIO as GPIO
from picamera import PiCamera
from picamera.array import PiRGBArray

# ---------------- GPIO SETUP ---------------- #
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor Pins
LEFT_EN, LEFT_IN1, LEFT_IN2 = 12, 5, 6
RIGHT_EN, RIGHT_IN3, RIGHT_IN4 = 13, 16, 20
MOTOR_PINS = [LEFT_EN, LEFT_IN1, LEFT_IN2, RIGHT_EN, RIGHT_IN3, RIGHT_IN4]

# Ultrasonic Sensor Pins
SENSORS = {
    "front": (17, 27),
    "left": (22, 23),
    "right": (24, 25),
    "back": (8, 7)
}

# Servo Pins
SERVO_DOOR = 18
SERVO_ARM = 19

# Setup
for pin in MOTOR_PINS + [SERVO_DOOR, SERVO_ARM]:
    GPIO.setup(pin, GPIO.OUT)

for trig, echo in SENSORS.values():
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)

# Motor PWM
left_pwm = GPIO.PWM(LEFT_EN, 1000)
right_pwm = GPIO.PWM(RIGHT_EN, 1000)
left_pwm.start(0)
right_pwm.start(0)

# Servo PWM
servo_door = GPIO.PWM(SERVO_DOOR, 50)
servo_arm = GPIO.PWM(SERVO_ARM, 50)
servo_door.start(0)
servo_arm.start(0)

# ---------------- MOTOR CONTROL ---------------- #
def set_motor(left_speed, right_speed):
    if left_speed >= 0:
        GPIO.output(LEFT_IN1, GPIO.HIGH)
        GPIO.output(LEFT_IN2, GPIO.LOW)
    else:
        GPIO.output(LEFT_IN1, GPIO.LOW)
        GPIO.output(LEFT_IN2, GPIO.HIGH)
    if right_speed >= 0:
        GPIO.output(RIGHT_IN3, GPIO.HIGH)
        GPIO.output(RIGHT_IN4, GPIO.LOW)
    else:
        GPIO.output(RIGHT_IN3, GPIO.LOW)
        GPIO.output(RIGHT_IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(abs(left_speed))
    right_pwm.ChangeDutyCycle(abs(right_speed))

def forward(speed=60): set_motor(speed, speed)
def backward(speed=60): set_motor(-speed, -speed)
def turn_left(speed=50): set_motor(-speed, speed)
def turn_right(speed=50): set_motor(speed, -speed)
def stop(): set_motor(0, 0)

# ---------------- ULTRASONIC ---------------- #
def measure_distance(trig, echo):
    GPIO.output(trig, False)
    time.sleep(0.0002)
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    pulse_start, pulse_end = 0, 0
    timeout = time.time() + 0.04
    while GPIO.input(echo) == 0 and time.time() < timeout:
        pulse_start = time.time()
    while GPIO.input(echo) == 1 and time.time() < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return distance

def read_all_sensors():
    distances = {}
    for key, (trig, echo) in SENSORS.items():
        distances[key] = measure_distance(trig, echo)
    return distances

# ---------------- SERVO CONTROL ---------------- #
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

# ---------------- CAMERA SETUP ---------------- #
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(1)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
min_area = 800

# ---------------- MAIN LOOP ---------------- #
def main():
    print("Starting Autonomous Trash Collector...")
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = bg_subtractor.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        cx = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                detected = True
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        distances = read_all_sensors()
        front, left, right, back = distances["front"], distances["left"], distances["right"], distances["back"]

        if front < 25:
            stop()
            print("Obstacle Ahead! Avoiding...")
            if left > right:
                turn_left()
            else:
                turn_right()
            time.sleep(0.5)
            stop()
        elif detected:
            center = img.shape[1] // 2
            offset = cx - center
            print("Trash detected, offset:", offset)
            if abs(offset) < 40:
                print("Moving forward...")
                forward(55)
            elif offset < 0:
                print("Turning left to align...")
                turn_left(45)
            else:
                print("Turning right to align...")
                turn_right(45)
            if front < 30:
                print("Collecting trash...")
                stop()
                open_collector()
                time.sleep(2)
                close_collector()
                backward(60)
                time.sleep(1)
                stop()
        else:
            print("Searching for trash...")
            turn_left(40)
            time.sleep(0.3)
            stop()

        rawCapture.truncate(0)
        time.sleep(0.1)

try:
    main()
except KeyboardInterrupt:
    print("Exiting Program...")
finally:
    stop()
    servo_door.stop()
    servo_arm.stop()
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()
    
