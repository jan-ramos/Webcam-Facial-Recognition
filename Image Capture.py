#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cv2
import os
from PIL import Image

image_path = "filepath"

cap = cv2.VideoCapture(0)
count = 0
max_images = 500

while True and count < max_images:
    ret, frame = cap.read()
    image = Image.fromarray(frame, 'RGB')
    image = image.resize((128,128))
    image.save(f"{image_path}/image-{count}.jpg")
    cv2.imshow("Image Taker",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count +=1

cap.release()
cv2.destroyAllWindows()

