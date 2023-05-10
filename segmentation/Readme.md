Overview:

Step 1:
Extract images from video clips.
- .mp4 files: Use video_to_images.py in this directory
- .asd files: Use ImageJ tool to extract images

Step 2:
Create 'ref' and 'test' folders with the images.
- ref: Select 3-10 images
- test: All images

Step 3:
Run unsupervised_img_segmentation_and_object_detection.py

- Input: Raw images (time series)
- Final Outputs: Metadata CSV file, cropped objects from each image

Step 4:
Run transfer_learning_clustering.py


