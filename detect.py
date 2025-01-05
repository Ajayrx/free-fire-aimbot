import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
with open("models/labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_enemies(image):
    """
    Detect enemies in the given image using YOLO.

    :param image: Input image (BGR format).
    :return: List of bounding boxes (x, y, w, h) and class IDs.
    """
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result_boxes = [boxes[i[0]] for i in indices]

    return result_boxes, class_ids

if __name__ == "__main__":
    # Test detection
    image = cv2.imread("test_image.jpg")  # Replace with a test image
    detected_boxes, ids = detect_enemies(image)

    for box in detected_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
