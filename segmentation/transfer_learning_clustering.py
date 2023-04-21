# Clustering using a transfer learning approach. A CNN (VGG16) is used to extract features. This is followed
# by dimensionality reduction using PCA (linear) and UMAP(non-linear). Then k-means/DBSCAN clustering is done 
#and clusters are visualized.


# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import umap
import umap.plot
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import seaborn as sns
from sklearn.pipeline import Pipeline
import os


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

# loop through each image in the dataset
for img in imgs:
        feat = extract_features(img,model)
        data[img] = feat

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape
feat = feat.reshape(-1,4096)
print(feat.shape)


# Define the pipeline
pipeline = Pipeline([
    ('pca', PCA(n_components=2, random_state=22)),
    ('umap_mapper', umap.UMAP(n_components=2, random_state=42)),
    ('kmeans_cluster', KMeans(n_clusters=5,random_state=22)),
    ('dbscan_cluster', DBSCAN(min_samples = 5))
])

# Fit the pipeline to the feature vectors
pipeline.fit(feat)

# Get the reduced feature vectors from PCA and UMAP
pca_out = pipeline.named_steps['pca'].fit_transform(feat)
umap_out = pipeline.named_steps['umap_mapper'].fit_transform(feat)

# Get the KMeans cluster labels for PCA and UMAP reduced feature vectors
pca_kmeans_labels = pipeline.named_steps['kmeans_cluster'].fit_predict(pca_out)
umap_kmeans_labels = pipeline.named_steps['kmeans_cluster'].fit_predict(umap_out)

# Get the DBSCAN cluster labels for PCA and UMAP reduced feature vectors
pca_dbscan_labels = pipeline.named_steps['dbscan_cluster'].fit_predict(pca_out)
umap_dbscan_labels = pipeline.named_steps['dbscan_cluster'].fit_predict(umap_out)

# create plots directory if it does not exist
if not os.path.exists('/jet/home/gkant/mAb1_20007#/plots'):
    os.makedirs('/jet/home/gkant/mAb1_20007#/plots')

# save PCA/kmeans plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_out[:, 0], y=pca_out[:, 1], hue=pca_kmeans_labels)
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/pca_kmeans.png', dpi=300, bbox_inches='tight')

# save UMAP/kmeans plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=umap_out[:, 0], y=umap_out[:, 1], hue=umap_kmeans_labels)
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/umap_kmeans.png', dpi=300, bbox_inches='tight')

# save PCA/DBSCAN plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_out[:, 0], y=pca_out[:, 1], hue=pca_dbscan_labels)
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/pca_dbscan.png', dpi=300, bbox_inches='tight')

# save UMAP/DBSCAN plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=umap_out[:, 0], y=umap_out[:, 1], hue=umap_dbscan_labels)
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/umap_dbscan.png', dpi=300, bbox_inches='tight')

     

'''
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
'''       
'''
# To identify no. of clusters using elbow method (pca)
pca = PCA(n_components = 2)
pca_out = pca.fit_transform(feat)

sse = []
list_k = list(range(2, 30))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(pca_out)
    
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/pca_kmeans_elbow_plt.png', dpi=300, bbox_inches='tight')

# To identify no. of clusters using elbow method (umap)
mapper = umap.UMAP(n_components=2)
umap_out = pca.fit_transform(feat)

sse = []
list_k = list(range(2, 30))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(umap_out)
    
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.savefig('/jet/home/gkant/mAb1_20007#/plots/umap_kmeans_elbow_plt.png', dpi=300, bbox_inches='tight')

'''
