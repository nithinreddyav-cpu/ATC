from picamera2 import Picamera2
import time

picam = Picamera2()
picam.configure(picam.create_preview_configuration())
picam.start()
time.sleep(2)
print("Camera OK!")
