import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Define the input and output folders
input_folder_path = "Input_Images/"
output_folder = 'Output_Images/'

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize counters for input and output images
input_image_count = 0
output_image_count = 0

# Confidence threshold for object detection (adjust as needed)
confidence_threshold = 0.5

# Loop through the images in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.JPG'):  # Assuming the images are in JPG format
        input_image_count += 1  # Increment the input image count
        current_image = input_image_count

        # Load and preprocess the input image
        input_image_path = os.path.join(input_folder_path, filename)
        image = Image.open(input_image_path)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            predictions = model(image_tensor)

        # Initialize variables to track the bounding boxes of all detected humans
        human_boxes = []

        # Process detections and find bounding boxes for all detected humans
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if score > confidence_threshold and label == 1:  # Class ID for 'person' is 1
                human_boxes.append(box.tolist())

        # Calculate the centroid of all detected human bounding boxes
        if human_boxes:
            centroid_x = sum((box[0] + box[2]) / 2 for box in human_boxes) / len(human_boxes)
            centroid_y = sum((box[1] + box[3]) / 2 for box in human_boxes) / len(human_boxes)

            # Calculate the width and height of the centered bounding box
            half_width = max(box[2] - box[0] for box in human_boxes) / 2
            half_height = max(box[3] - box[1] for box in human_boxes) / 2

            # Calculate the coordinates of the centered bounding box
            centered_box = (
                centroid_x - half_width,
                centroid_y - half_height,
                centroid_x + half_width,
                centroid_y + half_height
            )

            # Draw the centered bounding box on the image
            draw = ImageDraw.Draw(image)
            draw.rectangle(centered_box, outline='Yellow', width=25)
            draw.text((centered_box[0], centered_box[1]), f'Person', fill='red')

        # Save the annotated image in the output folder
        annotated_image_path = os.path.join(output_folder, filename)
        image.save(annotated_image_path)
        output_image_count += 1  # Increment the output image count

        print(f"{current_image} out of {input_image_count} processed")