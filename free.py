"""import cv2
import numpy as np

cap = cv2.VideoCapture("water.mp4")
ret, frame = cap.read()

# Select multiple ROIs manually
rois = cv2.selectROIs("Select Objects", frame, False, False)
cv2.destroyAllWindows()

# Store histogram and tracking windows for each ROI
roi_hists = []
track_windows = []

for roi in rois:
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    roi_hists.append(roi_hist)
    track_windows.append((x, y, w, h))

# Define termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for i in range(len(track_windows)):
        dst = cv2.calcBackProject([hsv], [0], roi_hists[i], [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_windows[i], term_crit)
        x, y, w, h = track_window
        track_windows[i] = track_window

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Tracked Objects', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

2
"""import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# User selects ROI interactively
x, y, w, h = cv2.selectROI("Select Object", frame, False, False)
cv2.destroyAllWindows()

# Extract ROI and convert to HSV
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram for tracking (higher bins for better accuracy)
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Define termination criteria for CamShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Apply CamShift for adaptive tracking
    ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)
    x, y, w, h = track_window

    # Draw a tracking box
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow('Tracking Moving Object', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

3
"""import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    cv2.imshow('Detected Moving Object', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

3
"""import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (e.g., marine species, coral)
classes = ["fish", "coral", "debris", "shark"]  # Modify as needed

# Open underwater video
cap = cv2.VideoCapture("water.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                x, y, w, h = (obj[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                label = f"{classes[class_id]} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Underwater Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

4
"""import cv2
import numpy as np

# Set absolute file paths for YOLO files
cfg_path = r"D:\project\yolov4.cfg"  # Update the path if needed
weights_path = r"D:\project\yolov4.weights"  # Update the path if needed
classes_path = r"D:\project\coco.names"  # List of object labels

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(classes_path, "r") as f:
    classes = f.read().splitlines()

# Open underwater video
cap = cv2.VideoCapture("water.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    
    # Convert frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]  # Ignore first 5 values (box parameters)
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Filter weak detections
                x, y, w, h = (obj[:4] * np.array([width, height, width, height])).astype(int)
                label = f"{classes[class_id]} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Underwater Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""
5
"""import cv2
import numpy as np

# Load YOLO model
cfg_path = "yolov4.cfg"
weights_path = "yolov4.weights"
classes_path = "coco.names"

net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(classes_path, "r") as f:
    classes = f.read().splitlines()

# Open video file or webcam
cap = cv2.VideoCapture("video.mp4")  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                x, y, w, h = (obj[:4] * np.array([width, height, width, height])).astype(int)
                label = f"{classes[class_id]} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

5

"""import cv2
import numpy as np

# Open underwater video
cap = cv2.VideoCapture("fish.mp4")
ret, frame = cap.read()

# Define the region of interest (ROI)
x, y, w, h = 200, 150, 50, 50  # Adjust for object position
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram for tracking
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup CamShift tracking
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Convert frame to HSV and apply back projection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Mask background subtraction with back projection
    dst = cv2.bitwise_and(dst, dst, mask=fgmask)

    # Apply CamShift tracking
    ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)
    x, y, w, h = track_window

    # Draw tracking rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Underwater Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

6

import cv2
import numpy as np

# Open underwater video
cap = cv2.VideoCapture("marine.mp4")
ret, frame = cap.read()

# Define the region of interest (ROI)
x, y, w, h = 200, 150, 50, 50  # Adjust for object position
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram for tracking
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup Mean Shift tracking criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Convert frame to HSV and apply back projection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Mask background subtraction with back projection
    dst = cv2.bitwise_and(dst, dst, mask=fgmask)

    # Apply Mean Shift tracking
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)
    x, y, w, h = track_window

    # Draw tracking rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Underwater Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()