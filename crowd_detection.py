import cv2
import torch
import numpy as np
import subprocess
from collections import deque
import os

# Step 1: Convert input video using ffmpeg (automated)
input_path = "/home/sanglap/customer_churn_prediction/Internshala/input_video.mp4"
converted_path = "/home/sanglap/customer_churn_prediction/Internshala/converted_video.mp4"
output_path = "/home/sanglap/customer_churn_prediction/Internshala/output_video.mp4"

print("Converting video using FFmpeg...")
ffmpeg_cmd = [
    "ffmpeg",
    "-i", input_path,
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-c:a", "aac",
    converted_path,
    "-y"  # Overwrite if exists
]
subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Conversion complete. Starting detection...")

# Step 2: Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4

# Step 3: Open converted video
cap = cv2.VideoCapture(converted_path)
ret, frame = cap.read()
if not ret:
    print("Failed to read")
    exit()
height, width = frame.shape[:2]

# Video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Step 4: Crowd Detection Setup
frame_count = 0
distance_threshold = 75
consecutive_frame_threshold = 10
group_buffer = deque(maxlen=consecutive_frame_threshold)
persistent_groups = {}
crowd_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model(frame)
    persons = [det for det in results.xyxy[0] if int(det[5]) == 0]

    centers = []
    for p in persons:
        x1, y1, x2, y2 = p[:4]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    close_group = []
    for i in range(len(centers)):
        group = [i]
        for j in range(len(centers)):
            if i != j:
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if dist < distance_threshold:
                    group.append(j)
        if len(set(group)) >= 3:
            close_group.append(tuple(sorted(set(group))))
    close_group = list(set(close_group))

    if close_group:
        group_buffer.append(close_group)
        for group in close_group:
            group_key = tuple(group)
            if group_key not in persistent_groups:
                persistent_groups[group_key] = 1
            else:
                persistent_groups[group_key] += 1

            if persistent_groups[group_key] == consecutive_frame_threshold:
                crowd_log.append([frame_count, len(group)])
                cv2.putText(frame, f"CROWD: {len(group)} people",
                            
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        group_buffer.clear()
        persistent_groups.clear()

    out.write(frame)

cap.release()
out.release()

# Step 5: Terminal summary
if crowd_log:
    print("\n Crowd Detected at:")
    for entry in crowd_log:
        print(f"Frame {entry[0]}: {entry[1]} people")
    print(f"\n Output video saved as '{output_path}'")
else:
    print("\n No crowd detected in the video.")
