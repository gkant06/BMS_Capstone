import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Set up the file path
path = '/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/'

# Set up the file paths for the folders containing the images
folder_paths = ["AmSO4_21223", "HELIX_21221", "MIDAS_Plus_21222"]

# Load the images from all folders and combine them into a single data set
try:
        os.mkdir("/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/combine")
except:
    print("Folder already found")
    
save_path = '/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/combine'

image_list = []
img_paths = []
for fpath in folder_paths:
    folder_path = path + fpath
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # image_list.append(img)
            img_path = fpath + '/' + filename
            img_paths.append(img_path)
            resized_img = cv2.resize(img, (512, 512)) # resize to 512 x 512
            image_list.append(resized_img)
            
            resized_img = resized_img.astype(np.uint8) # save as image in specific directory
            img = Image.fromarray(resized_img)
            #normalized_img = np.array(resized_img) / 255.0 # Normalize the pixel values
            img.save(os.path.join(save_path, fpath + '_' + filename))
            
            
            

data = {'image': image_list, 'path': img_paths}

df = pd.DataFrame(data)
df.to_csv('protein_cryst.csv')
            