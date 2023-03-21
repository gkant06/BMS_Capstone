# Once we obtain a segmented image- each object in an image is identified, cropped and saved as a new image in the output directory
# Create a metadata file with image name, other info

import cv2
import os
import csv
from datetime import datetime

# Define input and output directories
input_dir = 'input_images'
output_dir = 'output_images'
metadata_file = 'metadata.csv'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize metadata list
metadata = []

# Process each image in input directory
for filename in os.listdir(input_dir):
    # Check if file is an image
    if not filename.endswith(('png', 'jpg', 'jpeg')):
        continue
    
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(os.path.join(input_dir, filename))
    if image is None:
        print(f"Error: Could not load image {filename}")
        continue

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours, obtain bounding box, extract and save ROI
    object_number = 0
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        objects = original[y:y+h, x:x+w]
        output_filename = os.path.join(output_dir, 'object_{}_{}.png'.format(os.path.splitext(filename)[0], object_number))
        cv2.imwrite(output_filename, objects)
        object_number += 1

        # Append metadata for current ROI
        metadata.append([output_filename, filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (x, y, w, h)])

# Create metadata CSV file
with open(metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'parent_image', 'timestamp', 'object_location', 'object_count_in_parent_img'])
    
    # Count child images for each parent image and write metadata to CSV file
    for filename in os.listdir(input_dir):
        parent_image = filename
        child_image_count = 0
        for metadata_item in metadata:
            if metadata_item[1] == parent_image:
                writer.writerow([metadata_item[0], parent_image, metadata_item[2], metadata_item[3], child_image_count])
                child_image_count += 1
