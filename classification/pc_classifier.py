# for classifier
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Input, Activation
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# for loading and save images to folder
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Set up the file path
path = '/jet/home/thanhngp/BMS_classifer/ASO_Xstals_FInal_Time_Point/'

# Set up the file paths for the folders containing the images
folder_paths = ["AmSO4_21223", "HELIX_21221", "MIDAS_Plus_21222", 
                "MPD_21249", "Natrix_21225", "Nucleix_21224", 
                "PEG_Ionic_LQ_Rp_21355", "PEG_Ionic_LQ_21227",
                "PEG_pH_21228", "PEG_Rx_21230", "PEG400_ion_21229",
                "PEG_ion_21226", "pH_Clear_I_21245", "pH_Clear_II_21246",
                "Salt_Rx_21231", "Wizard_PEGion_1-4K_21232",
                "Wizard_PEGion_8-10K_21233"]
                
                
# Make predictions for each images
for fpath in folder_paths:
    folder_path = path + fpath
    if os.path.exists(folder_path):
        print("Found -->", folder_path)
    else:
        print("Did not find -->", folder_path)

# Create folders to contain well-formed & ill-formed droplets after classified
try:
        os.mkdir("/jet/home/thanhngp/BMS_classifer/ASO_good_droplet") 
except:
    print("Folder already found")
    
try:
        os.mkdir("/jet/home/thanhngp/BMS_classifer/ASO_bad_droplet")
except:
    print("Folder already found")
    
save_good = '/jet/home/thanhngp/BMS_classifer/ASO_good_droplet' # for storing well-formed droplets
save_bad = '/jet/home/thanhngp/BMS_classifer/ASO_bad_droplet'   # for storing ill-formed droplets

image_list = []
img_paths = []
img_names = []
screens = []
labels = []

# Specify image size and batch size
image_size = 224    
        
# Load classification model (trained VGG16)
model = keras.models.load_model('/jet/home/thanhngp/BMS_classifer/cnn-model')
model.summary()


# Make predictions for each images
for fpath in folder_paths:
    folder_path = path + fpath
    print(folder_path)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # add code for classify
            sample = image.load_img(img_path, target_size=(image_size, image_size))
            sample_array = image.img_to_array(sample) / 255.0 # convert the image to a numpy array and normalize the pixel values   
            sample_array = np.expand_dims(sample_array, axis=0) # add an extra dimension to the array to match the input shape expected by the model
            prediction = model.predict(sample_array)
            if prediction[0][0] <= 0.5:
                label = 1
            else:
                label = 0
            # add label to list
            labels.append(label) 
            
            # add image & path to list
            resized_img = cv2.resize(img, (512, 512)) # resize to 512 x 512
            image_list.append(resized_img)
            
            resized_img = resized_img.astype(np.uint8) # save as image in specific directory
            img = Image.fromarray(resized_img)
            # plt.imshow(img)
            
            if label == 1:
                save_path = os.path.join(save_good, fpath + '_' + filename)
            else:
                save_path = os.path.join(save_bad, fpath + '_' + filename)
            
            # save image to classified directory
            img.save(save_path)
            img_paths.append(save_path)
            
            # save sreen & image name
            screens.append(fpath)
            img_names.append(filename)
    
    # checkpoint after completed classify all images in one screen
    # save data to metafile
    data = {'path': img_paths, 'screen': screens, 'name': img_names, 'label': labels}
    #print(len(image_list))

    df = pd.DataFrame(data)
    df.to_csv('/jet/home/thanhngp/BMS_classifer/protein_cryst.csv')   
            
                
              
# save data to metafile

data = {'path': img_paths, 'screen': screens, 'name': img_names, 'label': labels}
#print(len(image_list))

df = pd.DataFrame(data)
df.to_csv('/jet/home/thanhngp/BMS_classifer/protein_cryst.csv') 