# Clustering using a transfer learning approach. A CNN (VGG16) is used to extract features. This is followed
# by dimensionality reduction using PCA. Then k-means clustering is done and clusters are visualized.


# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

path = '/jet/home/gkant/mAb1_20007#/output_images'
print(path)
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
imgs = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
          # adds only the image files to the flowers list
            imgs.append(file.name)
            
print(len(imgs))           
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = '/jet/home/gkant/mAb1_20007#/'

# lop through each image in the dataset
for img in imgs:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(img,model)
        data[img] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)


# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=4,random_state=22)
kmeans.fit(x)

# create a scatter plot of the reduced feature vectors, colored by their respective cluster labels
plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)

# add title and labels to the plot
plt.title("Clustering Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# show the plot
plt.show()

# function that lets you view a cluster (based on cluster label)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
   
# To identify no. of clusters using elbow method
#sse = []
#list_k = list(range(3, 10))

#for k in list_k:
#    km = KMeans(n_clusters=k, random_state=22)
#    km.fit(x)
    
#    sse.append(km.inertia_)

# Plot sse against k
#plt.figure(figsize=(6, 6))
#plt.plot(list_k, sse)
#plt.xlabel(r'Number of clusters *k*')
#plt.ylabel('Sum of squared distance');