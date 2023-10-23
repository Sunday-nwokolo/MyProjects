import tkinter as tk
from tkinter import Button, Label
import serial
import cv2
import os
import time

# Global variables
ser = serial.Serial('COM8', 9600)  # Replace 'COM_PORT' with your Arduino's port
cap = cv2.VideoCapture(0)

# Directory to save images
directory = "captured_images"
if not os.path.exists(directory):
    os.makedirs(directory)

def start_process():
    ser.write(b'START')
    while True:
        if ser.inWaiting():
            signal = ser.readline().decode('utf-8').strip()
            if signal == "CAPTURE":
                ret, frame = cap.read()
                if ret:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join(directory, f"image_{timestamp}.tiff")  # Changed extension to .tiff
                    cv2.imwrite(filename, frame)
                    print(f"Captured and saved image: {filename}")

# Create the main window
root = tk.Tk()
root.title("Image Capturing Control")

# Add a button to start the process
start_button = tk.Button(root, text="Start", command=start_process)
start_button.pack(pady=20)

root.mainloop()
