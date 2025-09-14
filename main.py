import os
import cv2
import numpy as np
from flask import Flask, send_from_directory

# Create the Flask app
app = Flask(__name__)

# =========================
# Object Detection Section
# =========================

# Create static directory if not exists
os.makedirs("static", exist_ok=True)

# Input video path (must be in your repo)
video_path = "input.mp4"

# Output video path inside static folder
output_path = "static/output.avi"

def run_object_detection():
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer (XVID codec, AVI format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Load class labels
    with open('voc.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load pre-trained MobileNet SSD model
    net = cv2.dnn.readNetFromCaffe(
        'models/deploy.prototxt',
        'models/mobilenet_iter_73000.caffemodel'
    )

    # Set backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Confidence threshold
    confidence_threshold = 0.5

    # Colors for classes
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    print("🚀 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished processing video")
            break

        (h, w) = frame.shape[:2]

        # Preprocess for the network
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, min(startX, w))
                startY = max(0, min(startY, h))
                endX = max(0, min(endX, w))
                endY = max(0, min(endY, h))

                label = f"{class_names[class_id]}: {confidence:.2f}"
                color = colors[class_id]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write processed frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎥 Output saved as {output_path}")

# Run detection once at startup
run_object_detection()

# =========================
# Flask Routes
# =========================

@app.route("/")
def home():
    return "✅ Object detection finished. Go to /download to get the video."

@app.route("/download")
def download():
    return send_from_directory("static", "output.avi", as_attachment=True)

# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port)
