import pandas as pd

filepath = '/jet/home/gkant/mAb1_20007#/'

# Read the CSV file
df = pd.read_csv(filepath+'metadata.csv')

# Sort the DataFrame by image name
df = df.sort_values('parent_image')

# Create a dictionary to map image names to frame numbers
frame_numbers = {}
frame_count = 1
for image_name in df['parent_image']:
    if image_name not in frame_numbers:
        frame_numbers[image_name] = frame_count
        frame_count += 1

# Create a new column in the DataFrame containing the frame numbers
df['frame_number'] = df['parent_image'].map(frame_numbers)

# Write the updated DataFrame to a new CSV file
df.to_csv(filepath+'metadata_updated.csv', index=False)