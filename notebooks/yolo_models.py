# ensemble_yolov8_detection.py
# Sample Voting-Based Ensemble YOLOv8 Detection Framework
# Reference implementation for underwater fish detection

from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------------------------------
# Load Multiple YOLOv8 Models
# ---------------------------------------------------

model_n = YOLO("YOLOv8n.pt")
model_s = YOLO("YOLOv8s.pt")
model_m = YOLO("YOLOv8m.pt")
model_l = YOLO("YOLOv8l.pt")

models = [model_n, model_s, model_m, model_l]

# ---------------------------------------------------
# Load Image
# ---------------------------------------------------

image_path = "test_image.jpg"
image = cv2.imread(image_path)

# ---------------------------------------------------
# Store All Predictions
# ---------------------------------------------------

all_boxes = []
all_scores = []
all_classes = []

# ---------------------------------------------------
# Run Inference Using Multiple Models
# ---------------------------------------------------

for model in models:

    results = model(image)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):

            all_boxes.append(box)
            all_scores.append(float(score))
            all_classes.append(int(cls))

# ---------------------------------------------------
# Convert Boxes for NMS
# ---------------------------------------------------

nms_boxes = []

for box in all_boxes:

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    nms_boxes.append([int(x1), int(y1), int(width), int(height)])

# ---------------------------------------------------
# Apply Non-Maximum Suppression (NMS)
# ---------------------------------------------------

indices = cv2.dnn.NMSBoxes(
    nms_boxes,
    all_scores,
    score_threshold=0.4,
    nms_threshold=0.5
)

# ---------------------------------------------------
# Draw Final Ensemble Predictions
# ---------------------------------------------------

for i in indices:

    i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i

    x, y, w, h = nms_boxes[i]

    confidence = all_scores[i]
    class_id = all_classes[i]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    label = f"Fish {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# ---------------------------------------------------
# Save Output
# ---------------------------------------------------

output_path = "ensemble_output.jpg"

cv2.imwrite(output_path, image)

print("Ensemble detection completed successfully.")
print(f"Output saved at: {output_path}")
