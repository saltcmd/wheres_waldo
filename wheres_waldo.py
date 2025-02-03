import cv2
import os
import numpy as np

# Paths to configuration, weights, and class names
weights_path = "./waldo.weights"
config_path = "./yolov4-tiny.cfg"
names_path = "./obj.names"
image_dir = "./samples"  # Path to test image directory

# Define target size for training proportions
TRAINING_WIDTH = 3500
TRAINING_HEIGHT = 2400
TILE_SIZE = 416  # Size of YOLO input tiles

# Define screen dimensions for scaling the final display image
SCREEN_WIDTH = 1920  # Replace this with your screen width
SCREEN_HEIGHT = 1080  # Replace this with your screen height

# Load YOLOv4-tiny model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to perform detection on a single tile
def detect_objects(image_tile, offset_x=0, offset_y=0, scale=1.0):
    height, width, _ = image_tile.shape
    blob = cv2.dnn.blobFromImage(image_tile, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Initialize lists for storing detection results
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Adjust the confidence threshold
            if confidence > 0.2:  # Minimum confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2)) + offset_x
                y = int(centerY - (h / 2)) + offset_y

                # Scale back to the original image size
                boxes.append([int(x / scale), int(y / scale), int(w / scale), int(h / scale)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Function to dynamically tile an image
def dynamic_tiling(image, tile_size=416, overlap_ratio=0.5):
    height, width, _ = image.shape
    stride = int(tile_size * (1 - overlap_ratio))  # Calculate stride for overlap
    tiles = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            # Extract tile and its offsets
            tile = image[y:y_end, x:x_end]
            tiles.append((tile, x, y))
    return tiles

# Function to resize an image to fit the training proportions (3500x2400)
def resize_to_training_scale(image, target_width=3500, target_height=2400):
    height, width, _ = image.shape
    scale_width = target_width / width
    scale_height = target_height / height
    scale = min(scale_width, scale_height)  # Choose the smaller scaling factor to maintain aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scale

# Function to resize an image to fit the screen for display
def resize_to_fit_screen(image, screen_width, screen_height):
    height, width = image.shape[:2]
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)  # Choose the smaller scaling factor to fit within the screen
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Main detection function
def detect_objects_in_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    original_image = image.copy()

    # Resize image to match training proportions
    image, scale = resize_to_training_scale(image, TRAINING_WIDTH, TRAINING_HEIGHT)
    print(f"Resized image with scale factor: {scale}")

    # Dynamic tiling for the resized image
    tiles = dynamic_tiling(image, TILE_SIZE, overlap_ratio=0.5)

    # Variables to track the single highest confidence detection
    highest_confidence = 0
    best_detection = None  # To store the best detection

    # Process each tile
    for tile, offset_x, offset_y in tiles:
        boxes, confidences, class_ids = detect_objects(tile, offset_x, offset_y, scale)

        # Find the highest-confidence detection
        for i in range(len(confidences)):
            if confidences[i] > highest_confidence and classes[class_ids[i]] == "class1":  # Assuming "class1" is Waldo
                highest_confidence = confidences[i]
                best_detection = {
                    "box": boxes[i],
                    "confidence": confidences[i]
                }

    # Draw only the highest-confidence detection on the original image
    if best_detection:
        box = best_detection["box"]
        confidence = best_detection["confidence"]
        x, y, w, h = box
        label = f"Waldo: {confidence:.2f}"
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize the image to fit the screen
        resized_image = resize_to_fit_screen(original_image, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Display the image with the detection
        cv2.imshow("Highest-Confidence Detection", resized_image)
        cv2.waitKey(0)
    else:
        print(f"Waldo not found in {image_path}.")

# Process all images in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    print(f"Processing {image_path}...")
    detect_objects_in_image(image_path)

# Close all Windows
cv2.destroyAllWindows()
