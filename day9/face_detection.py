import urllib.request
url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
urllib.request.urlretrieve(url, "detector.tflite")
print("Downloaded!")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup detector
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect faces
    results = detector.detect(mp_image)

    # Draw bounding boxes
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            cv2.rectangle(
                frame,
                (bbox.origin_x, bbox.origin_y),
                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                (0, 255, 0), 2
            )
            # Show confidence score
            score = round(detection.categories[0].score, 2)
            cv2.putText(frame, f"Face: {score}", 
                       (bbox.origin_x, bbox.origin_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()