# For commands
import glob
import os
#os.chdir('/content/')
from ipywidgets import HTML, VBox
from plotly import graph_objects as go

import time
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
# For array manipulation
import numpy as np
import pandas as pd
import pandas.util.testing as tm
# For visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import cv2
import imageio as io
from pylab import *
from sklearn.preprocessing import StandardScaler
import umap as UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#For model performance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import joblib
#For model training
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
            

class AEImageClustering:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.trained = False
        self.df = None
        model_path = os.path.join(image_dir,'encoder_model.h5')
        self.train_files, self.test_files = self.read_images(image_dir)
        self.train_data = self.image2array(self.train_files)
        print("Length of training dataset:",self.train_data.shape)
        self.test_data = self.image2array(self.test_files)
        print("Length of test dataset:",self.test_data.shape)
        
        if os.path.isfile(model_path):
            self.model = load_model(model_path)
            tf.keras.utils.plot_model(self.model, to_file='model.png')
            optimizer = Adam(learning_rate=0.001) 
            self.model.compile(optimizer=optimizer, loss='mse')
            self.trained = True
        else:
            self.model = self.encoder_decoder_model()
            tf.keras.utils.plot_model(self.model, to_file='model.png')
            optimizer = Adam(learning_rate=0.001) 
            self.model.compile(optimizer=optimizer, loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=6,min_delta=0.0001) 
            checkpoint = ModelCheckpoint(os.path.join(image_dir,'encoder_model.h5'), monitor='val_loss', mode='min', save_best_only=True) 
            self.model.fit(self.train_data, self.train_data, epochs=35, batch_size=32,validation_data=(self.test_data,self.test_data),callbacks=[early_stopping,checkpoint])
        #self.model.summary()

    def get_features(self, layer):
        d = np.concatenate([self.train_data,self.test_data],axis=0)
        X_encoded = []
        i=0
        # Iterate through the full training set.
        for batch in self.get_batches(d, batch_size=300):
            i+=1
            # This line runs our pooling function on the model for each batch.
            X_encoded.append(self.feature_extraction(self.model, batch,layer=layer))
            
        X_encoded = np.concatenate(X_encoded)
        X_encoded = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
        scaled_features = StandardScaler().fit_transform(X_encoded)
        umap_reducer = UMAP.UMAP(random_state=142)
        print('calculating umap space..')
        umap_values = umap_reducer.fit_transform(scaled_features)
        #lisp=train_files
        #lisp.extend(test_files)
        transform = TSNE 
        trans = transform(n_components=2) 
        values = trans.fit_transform(X_encoded) 
        transform = PCA 
        trans = transform(n_components=2) 
        PCAvalues = trans.fit_transform(X_encoded) 
        return values, PCAvalues, umap_values
        
    def get_batches(self, data, batch_size=1000):
        """
        Taking batch of images for extraction of images.
        Arguments:
        data - (np.ndarray or list) - list of image array to get extracted features.
        batch_size - (int) - Number of images per each batch
        Returns:
        list - extracted features of each images
        """

        if len(data) < batch_size:
            return [data]
        n_batches = len(data) // batch_size
        
        # If batches fit exactly into the size of df.
        if len(data) % batch_size == 0:
            return [data[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]   

        # If there is a remainder.
        else:
            return [data[i*batch_size:min((i+1)*batch_size, len(data))] for i in range(n_batches+1)]

    def get_df(self, n_clusters=12,force=False, layer=12):
        if self.df is None or force:
            tsne, pca, umap_dimensions = self.get_features(layer=layer)
            lisp=np.concatenate([self.train_files,self.test_files],axis=0)
            self.df = pd.DataFrame ({'image_name':lisp})
            self.df['UMAP Dimension 1'] = umap_dimensions[:,0]
            self.df['UMAP Dimension 2'] = umap_dimensions[:,1]
            self.df['tsne1'] = tsne[:,0]
            self.df['tsne2'] = tsne[:,1]
            self.df['PCA1'] = pca[:,0]
            self.df['PCA2'] = pca[:,1]
            kmeans = KMeans(n_clusters = n_clusters, random_state=0).fit(tsne)
            labels=kmeans.labels_
            centroids = kmeans.cluster_centers_
            self.df['tsne_cluster_labels'] = labels
            kmeans = KMeans(n_clusters = n_clusters, random_state=0).fit(pca)
            labels=kmeans.labels_
            centroids = kmeans.cluster_centers_
            self.df['pca_cluster_labels'] = labels
            kmeans = KMeans(n_clusters = n_clusters, random_state=0).fit(umap_dimensions)
            labels=kmeans.labels_
            centroids = kmeans.cluster_centers_
            self.df['umap_cluster_labels'] = labels
        return self.df

    def interactive_plot(self, df, fig, template, event="hover") :
        """
        Make a plot react on hover or click of a data point and update a HTML preview below it.
        **template** Should be a string and contain placeholders like {colname} to be replaced by the value
        of the corresponding data row.
        
        """
    
        html = HTML("")
    
        def update(trace, points, state):
            ind = points.point_inds[0]
            row = df.loc[ind].to_dict()
            html.value = template.format(**row)
            #print(template.format(**row))
    
        fig = go.FigureWidget(data=fig.data, layout=fig.layout)
    
        if event == "hover" :
            fig.data[0].on_hover(update)
        else :
            fig.data[0].on_click(update)
    
        return fig,VBox([fig, html])
    
    def plot_thumb(self,x,y1,y2,row,col,ind,title,xlabel,ylabel,label,isimage=False,color='r', cmap='gray'):

        """
        This function is used for plotting images and graphs (Visualization of end results of model training)
        Arguments:
        x - (np.ndarray or list) - an image array
        y1 - (list) - for plotting graph on left side.
        y2 - (list) - for plotting graph on right side.
        row - (int) - row number of subplot
        col - (int) - column number of subplot
        ind - (int) - index number of subplot
        title - (string) - title of the plot 
        xlabel - (list) - labels of x axis
        ylabel - (list) - labels of y axis
        label - (string) - for adding legend in the plot
        isimage - (boolean) - True in case of image else False
        color - (char) - color of the plot (prefered green for training and red for testing).
        """

        plt.subplot(row,col,ind)
        if isimage:
            plt.imshow(x, cmap=cmap)
            plt.title(title)
            plt.axis('off')
        else:
            plt.plot(y1,label=label,color='g'); plt.scatter(x,y1,color='g')
            if y2!='': plt.plot(y2,color=color,label='validation'); plt.scatter(x,y2,color=color)
            plt.grid()
            plt.legend()
            plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
        
    def plot(self,x='tsne1',y='tsne2',color='tsne_cluster_labels', df=None, n_clusters=None, force=False):
        import plotly.express as px
        if df is None:
            df = self.get_df(n_clusters=n_clusters, force=force)
        template="<img src='{image_name}' height=200 width=200>"
        #print(template)
        fig = px.scatter(df,x=x,y=y,
                         color=color,
                         color_discrete_sequence=px.colors.qualitative.Dark24,
                         #render_mode='webgl',
                         hover_data=['image_name'])
        fig,vbox=self.interactive_plot(df, fig, template)
        display(vbox)

    def feature_extraction(self, model, data, layer = 4):
    
        """
        Creating a function to run the initial layers of the encoder model. (to get feature extraction from any layer of the model)
        Arguments:
        model - (Auto encoder model) - Trained model
        data - (np.ndarray) - list of images to get feature extraction from trained model
        layer - (int) - from which layer to take the features(by default = 4)
        Returns:
        pooled_array - (np.ndarray) - array of extracted features of given images
        """
    
        encoded = K.function([model.layers[0].input],[model.layers[layer].output])
        encoded_array = encoded([data])[0]
        pooled_array = encoded_array.max(axis=-1)
        return encoded_array


    def encoder_decoder_model(self):
        """
        Used to build Convolutional Autoencoder model architecture to get compressed image data which is easier to process.
        Returns:
        Auto encoder model
        """
        #Encoder 
        model = Sequential(name='Convolutional_AutoEncoder_Model')
        model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 3),padding='same', name='Encoding_Conv2D_1'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_1'))
        model.add(Conv2D(128, kernel_size=(3, 3),strides=1,activation='relu',padding='same', name='Encoding_Conv2D_2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_2'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same', name='Encoding_Conv2D_3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_3'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',padding='same', name='Encoding_Conv2D_4'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2,padding='valid', name='Encoding_MaxPooling2D_4'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Encoding_Conv2D_5'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
        
        #Decoder
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Decoding_Conv2D_1'))
        model.add(UpSampling2D((2, 2), name='Decoding_Upsamping2D_1'))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='Decoding_Conv2D_2'))
        model.add(UpSampling2D((2, 2), name='Decoding_Upsamping2D_2'))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',name='Decoding_Conv2D_3'))
        model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_3'))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',name='Decoding_Conv2D_4'))
        model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_4'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',name='Decoding_Conv2D_5'))
        model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_5'))
        model.add(Conv2D(3, kernel_size=(3, 3), padding='same',activation='sigmoid',name='Decoding_Output'))
        return model

    def image2array(self, file_array):
        """
        Reading and Converting images into numpy array by taking path of images.
        Arguments:
        file_array - (list) - list of file(path) names
        Returns:
        A numpy array of images. (np.ndarray)
        """
        image_array = []
        for path in tqdm(file_array):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            image_array.append(np.array(img))
        image_array = np.array(image_array)
        image_array = image_array.reshape(image_array.shape[0], 224, 224, 3) 
        image_array = image_array.astype('float32')
        image_array /= 255 
        return np.array(image_array)

    def read_images(self, image_dir):
        types = ('*.JPG','*.jpg','*.PNG','*.png')
        file_path = []
        for ftype in types:
            file_path.extend(glob.iglob(image_dir+'/'+ftype))
        print('Found {} files'.format(len(file_path)))
        #file_path = [f for f in glob.iglob(image_dir+'/*.JPG')]
        train_files, test_files = train_test_split(file_path, test_size = 0.15)
        return train_files, test_files
