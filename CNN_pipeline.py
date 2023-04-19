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

import os

from PIL import Image

import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# specify image size and batch size
image_size = 224
batch_size = 32

# specify train directory
train_dir = '/jet/home/thanhngp/BMS_classifer/Protein_Crystalization_train'
test_dir = '/jet/home/thanhngp/BMS_classifer/Protein_Crystalization_test'       
        
# Data Augmentation
image_generator = ImageDataGenerator(
        rescale=1/255)
        #rotation_range=10, # rotation
        #horizontal_flip=True, # horizontal flip
        #brightness_range=[0.8,1.2])# brightness)

#Train & Validation Split
train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(image_size, image_size),
                                                 color_mode='rgb',
                                                 class_mode='categorical', # update class_mode to categorical
                                                 classes=['0', '1'])

validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=test_dir,
                                                 shuffle=True,
                                                 target_size=(image_size, image_size),
                                                 color_mode='rgb',
                                                 class_mode='categorical', # update class_mode to categorical
                                                 classes=['0', '1'])

sample_test_dir = '/jet/home/thanhngp/BMS_classifer/sample'

test_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=sample_test_dir,
                                                 target_size=(image_size, image_size),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 classes=['0', '1'])
                                                 
def plot_sample(train_data):
    plt.figure(figsize=(10, 5))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        img = train_data[i][0][0]  # Extract the image from the batch
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        label = train_data[i][1][0]  # Extract the label from the batch
        ax.imshow(img)
        ax.set_title(f"Label: {int(label[1])}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_classification_data.png')
    plt.show()

# Call the function with the train_dataset
plot_sample(train_dataset)

################################################

# Define the models
def cnn_model(train_data, val_data, layers ,epochs, optimizer, loss, metrics, image_size):
    if layers == 'custom':
        model = Sequential()
        # Add layers to the model
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    elif layers == 'vgg':
        #model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        # Load VGG16 model (excluding top layers)
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
        model = Sequential()
        # Add VGG16 layers to the model
        model.add(vgg16)
        
    elif layers == 'resnet':
        # Load ResNet50 model (excluding top layers)
        resnet = ResNet50(pretrained=True, include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
        model = Sequential()
        # Add ResNet50 layers to the model
        model.add(resnet)
        # Add global average pooling layer
        #model.add(GlobalAveragePooling2D())

    
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    
    model.compile(optimizer = optimizer,
                 loss = loss,
                 metrics = metrics)
    
    #callback = keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                            patience=5,
    #                                            restore_best_weights=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [early_stopping]
    
    model.summary()
    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks)
    model.save('/jet/home/thanhngp/BMS_classifer/cnn-model')
    
    loss, accuracy = model.evaluate(validation_dataset)
    train_accuracy = pd.DataFrame(history.history['accuracy'])
    train_accuracy.to_csv('train_acc.csv', index=True)
    val_accuracy = pd.DataFrame(history.history['val_accuracy'])
    val_accuracy.to_csv('val_acc.csv', index=True)
    
    # plot training curve
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy"])
    plt.savefig('resnet_acc.png')
    plt.show()


# define metrics
#layers = 'custom' # 4 convolutional layers
#layers = 'vgg'
layers = 'resnet'
epochs = 30
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['accuracy']
cnn_model(train_dataset, validation_dataset, layers ,epochs, optimizer, loss, metrics, image_size)

# Predict test data
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

# Iterate through all batches in the test dataset
for i in range(len(test_dataset)):
    # Get a batch of images and their ground truth labels
    x, y = test_dataset[i]
    # Generate predictions for the batch
    batch_pred = (model.predict(x) > 0.5).astype(int)
    # Append the batch predictions and ground truth labels to the lists
    y_pred.append(batch_pred)
    y_true.append(y)
    
# Concatenate the predictions and ground truth labels for all batches
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

#print(y_pred)
#print(y_true)

auc_roc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
print('Weighted-averaged AUC-ROC score:', auc_roc_weighted)
# custom AUC = 0.8799
#    vgg AUC = 0.9
# resnet AUC = 0.61
