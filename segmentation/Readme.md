Overview:

Step 1:
Extract images from video clips.
- .mp4 files: Use video_to_images.py in this directory
- .asd files: Use ImageJ tool to extract images

Step 2:
Create 'ref' and 'test' folders with the images.
- ref: Select 3-10 images based on total number of images
- test: All images

Step 3:
Run unsupervised_img_segmentation_and_object_detection.py

- Input: Raw images (time series)
- Final Outputs: Metadata CSV file, cropped objects from each image

Step 4:
Run transfer_learning_clustering.py

- Performs feature extraction using CNN, dimensionality reduction using PCA/UMAP and k-means clustering
- Output plots are saved in a new folder called 'Plots'
- Metadata file is modified to include cluster IDs from both methods

Step 5 (Optional. However, required to generate time series plots shown in next step or for analysis on time series data):
Run add_frame_number_metadata.py

- Adds a column for frame number to metadata CSV file

Step 6:
Run Time_series_analysis.ipynb

- To generate frame by frame analysis plots



