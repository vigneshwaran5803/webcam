import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Load the class labels
class_names = []
with open('voc.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

# Set preferred backend and target for faster inference (optional)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Confidence threshold to filter weak detections
confidence_threshold = 0.5

# Colors for different classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

print("Starting object detection. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Get the frame dimensions
    (h, w) = frame.shape[:2]

    # Preprocess the frame for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Get the class index
            class_id = int(detections[0, 0, i, 1])

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            startX = max(0, min(startX, w))
            startY = max(0, min(startY, h))
            endX = max(0, min(endX, w))
            endY = max(0, min(endY, h))

            # Draw the prediction on the frame
            label = f"{class_names[class_id]}: {confidence:.2f}%"
            color = colors[class_id]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Display the label at the top of the bounding box
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
