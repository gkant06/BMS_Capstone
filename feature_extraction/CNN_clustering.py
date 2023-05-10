# Clustering using a transfer learning approach. A CNN (VGG16) is used to extract features. This is followed
# by dimensionality reduction using PCA (linear) and UMAP(non-linear). Then k-means clustering is done 
#and clusters are visualized.


# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
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
import pandas as pd

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
    ('kmeans_cluster_pca', KMeans(n_clusters=5,random_state=22)),
    ('kmeans_cluster_umap', KMeans(n_clusters=2,random_state=22))
])

# Fit the pipeline to the feature vectors
pipeline.fit(feat)

# Get the reduced feature vectors from PCA and UMAP
pca_out = pipeline.named_steps['pca'].fit_transform(feat)
umap_out = pipeline.named_steps['umap_mapper'].fit_transform(feat)

# Get the KMeans cluster labels for PCA and UMAP reduced feature vectors
pca_kmeans_labels = pipeline.named_steps['kmeans_cluster_pca'].fit_predict(pca_out)
umap_kmeans_labels = pipeline.named_steps['kmeans_cluster_umap'].fit_predict(umap_out)

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


# holds the cluster id for pca, umap and the images
groups = {}
for file, cluster_pca, cluster_umap in zip(filenames, pca_kmeans_labels, umap_kmeans_labels):
    if cluster_pca not in groups.keys():
        groups[cluster_pca] = {"pca": [], "umap": []}
    groups[cluster_pca]["pca"].append(file)

    if cluster_umap not in groups.keys():
        groups[cluster_umap] = {"pca": [], "umap": []}
    groups[cluster_umap]["umap"].append(file)


# function that lets you view clusters (based on cluster label)        
def view_cluster(cluster, plot_type="pca", save_dir=None):

    # create subdirectory for cluster if save_dir is specified
    if save_dir:
        plot_dir = save_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    plt.figure(figsize=(20,20))
    # gets the list of filenames for a cluster
    files = groups.get(cluster, {}).get(plot_type, [])
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # create subplot grid
    num_rows = 6
    num_cols = 5
    num_images = min(len(files), num_rows * num_cols)
    for index in range(num_images):
        plt.subplot(num_rows, num_cols, index+1)
        img = load_img(files[index])
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    # save figure if save_dir is specified
    if save_dir:
        plot_path = os.path.join(plot_dir, f"cluster_{cluster}.png")
        plt.savefig(plot_path)
    plt.show()

# Cluster visualization
for cluster in set(pca_kmeans_labels):
    view_cluster(cluster, plot_type="pca", save_dir='/jet/home/gkant/mAb1_20007#/plots/cluster_pca_kmeans/')
    
for cluster in set(umap_kmeans_labels):
    view_cluster(cluster, plot_type="umap", save_dir='/jet/home/gkant/mAb1_20007#/plots/cluster_umap_kmeans/')


# Read in the metadata file as a DataFrame
metadata = pd.read_csv("/jet/home/gkant/mAb1_20007#/metadata.csv")

# Iterate through the file paths in the metadata file
pca_labels = []
umap_labels = []
for filepath in metadata["image"]:
    # Extract the filename from the filepath
    filename = os.path.basename(filepath)

    # Look up the corresponding PCA and UMAP cluster labels in the groups dictionary
    pca_label = None
    umap_label = None
    for cluster_label, cluster_data in groups.items():
        if filename in cluster_data["pca"]:
            pca_label = cluster_label
        if filename in cluster_data["umap"]:
            umap_label = cluster_label

    # Add the PCA and UMAP cluster labels to the lists
    pca_labels.append(pca_label)
    umap_labels.append(umap_label)

# Add the PCA and UMAP cluster labels as new columns to the metadata DataFrame
metadata["pca_cluster_label"] = pca_labels
metadata["umap_cluster_label"] = umap_labels

# Write the updated metadata file back to disk
metadata.to_csv("/jet/home/gkant/mAb1_20007#/metadata.csv", index=False)


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
