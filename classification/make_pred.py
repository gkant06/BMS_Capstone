import matplotlib.pyplot as plt
#from sklearn.datasets import make_circles
#from sklearn.model_selection import train_test_split
import time

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Input, Activation
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import os

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# specify image size and batch size
image_size = 224
batch_size = 1      
        
# Data Augmentation
image_generator = ImageDataGenerator(
        rescale=1/255)
model = keras.models.load_model('/jet/home/thanhngp/BMS_classifer/cnn-model')
model.summary()

sample_test_dir = '/jet/home/thanhngp/BMS_classifer/sample'

test_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=sample_test_dir,
                                                 target_size=(image_size, image_size),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 classes=['0', '1'])

y_pred = []
y_true = []

#def plot_sample(train_data):
#    plt.figure(figsize=(10, 5))
#    for i in range(8):
#        ax = plt.subplot(2, 4, i + 1)
#        img = train_data[i][0][0]  # Extract the image from the batch
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#        label = train_data[i][1][0]  # Extract the label from the batch
#        ax.imshow(img)
#        ax.set_title(f"Label: {int(label[1])}")
#        ax.axis('off')
#    plt.tight_layout()
#    plt.savefig('sample_classification_data.png')
#    plt.show()

# Call the function with the train_dataset
#plot_sample(train_dataset)

# Iterate through all batches in the test dataset
for i in range(len(test_dataset)):
    # Get a batch of images and their ground truth labels
    x, y = test_dataset[i]
    # Generate predictions for the batch
    batch_pred = (model.predict(x) > 0.5).astype(int)#[:,0]
    
    #x = cv2.cvtColor(x[0][0], cv2.COLOR_BGR2RGB)
    #plt.imshow(x[0][0])
    # Append the batch predictions and ground truth labels to the lists
    y_pred.append(batch_pred)
    y_true.append(y)
    
# Concatenate the predictions and ground truth labels for all batches
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

print('Pred:', y_pred)
print('True:', y_true)

auc_roc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
print('Weighted-averaged AUC-ROC score:', auc_roc_weighted)
