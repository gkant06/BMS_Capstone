# Extracting images from a video for a single video

#Using OpenCV
import cv2

# open the video file
video = cv2.VideoCapture('mAb1_20019#_C1_C1_0-129.mp4')

# get the frame rate of the video - fps (frames per second)
fps = int(video.get(cv2.CAP_PROP_FPS))

# initialize a counter for the image filenames
count = 0

# loop through the video frames
while True:
    # read a frame from the video
    ret, frame = video.read()
    
    # if the frame was not read successfully, break the loop
    if not ret:
        break
    
    # save the frame as an image
    filename = f'image_{count:03d}.jpg'
    cv2.imwrite(filename, frame)
    
    # increment the counter
    count += 1
    
    # move to the next frame
    video.set(cv2.CAP_PROP_POS_FRAMES, count * fps)
