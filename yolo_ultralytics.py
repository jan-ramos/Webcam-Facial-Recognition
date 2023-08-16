#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
names = model.names
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    if results[0].boxes.id is None:
            pass
    else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
    if results[0].boxes.cls is None:
            pass
    else:
            cls = results[0].boxes.cls.cpu().numpy().astype(int)
    
    for box, id,cl in zip(boxes, ids,cls):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {names[cl]}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

