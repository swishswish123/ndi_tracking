import cv2
import os

folder_name = 'images'

cap = cv2.VideoCapture(0)

count = 0

if not os.path.exists(folder_name):
  os.makedirs(folder_name)

while True:
  success, image = cap.read()
  cv2.imwrite(f"images/{count}.png", image)
  count += 1
