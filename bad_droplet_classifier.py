import matplotlib.pyplot as plt
#from sklearn.datasets import make_circles
#from sklearn.model_selection import train_test_split
import time

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

from PIL import Image

import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# prepare images as input to CNN model
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# specify image size and batch size
image_size = 512
batch_size = 20

# specify train directory
train_dir = '/jet/home/thanhngp/BMS_classifer/Protein_Crystalization_train'
test_dir = '/jet/home/thanhngp/BMS_classifer/Protein_Crystalization_test'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='binary')
        
start = time.time()


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
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

train_steps_per_epoch = 10
test_steps_per_epoch = 10 
epochs = 35
history = model.fit(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_steps_per_epoch)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps_per_epoch)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

# store train and validation accuracy for plotting
train_accuracy = pd.DataFrame(history.history['accuracy'])
train_accuracy.to_csv('train_acc.csv', index=True)
val_accuracy = pd.DataFrame(history.history['val_accuracy'])
val_accuracy.to_csv('val_acc.csv', index=True)

end = time.time()
print('time taken:', end - start)




# make and store predictions
from tensorflow.keras.preprocessing import image

# load prediction data info
#sample_file = '/jet/home/thanhngp/HW2-CNN/sample_submission.csv'
#sample_label = pd.read_csv( sample_file )
# print('There are ', sample_label.shape[0], ' testing images in ', len(sample_label.label.unique()), ' classes.')
sample_test_dir = '/jet/home/thanhngp/BMS_classifer/sample'

image_size = 512
img_path = []
labels = []
for f in os.listdir(sample_test_dir):
    path = os.path.join(sample_test_dir,f)
    img = cv2.imread(path)
    if img is not None:
        img_path.append(f)
        print(path)
        sample = image.load_img(path, target_size=(image_size, image_size))
    
        sample_array = image.img_to_array(sample) / 255.0 # convert the image to a numpy array and normalize the pixel values
    
        sample_array = np.expand_dims(sample_array, axis=0) # add an extra dimension to the array to match the input shape expected by the model
    
        prediction = model.predict(sample_array)
        
    
        if prediction >= 0.5:
            label = 1
        else:
            label = 0
        labels.append(label)
        print(label)
        
data = {'image_path': img_path, 'label': labels}
print(len(img_path))

df = pd.DataFrame(data)
df.to_csv('/jet/home/thanhngp/BMS_classifer/good_droplets.csv')






