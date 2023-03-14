from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics

import datetime
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint

class rVAE:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_model()

    def build_encoder(self):
        input_img = Input(shape=self.input_shape)
        l = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        l1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(l)
        l2 = Conv2D(64, (3, 3), activation='relu', padding='same')(l1)
        l3 = Conv2D(128, (3, 3), activation='relu', padding='same')(l2)
        shape_before_flattening = K.int_shape(l3)
        print(shape_before_flattening)
        flatten = Flatten()(l3)
        d1 = Dense(32, activation='relu')(flatten)
        z_mean = Dense(self.latent_dim)(d1)
        z_log_var = Dense(self.latent_dim)(d1)
        
        #epsilon = Lambda(lambda x:K.random_normal(shape=(K.shape(x)[0], self.latent_dim), mean=0., stddev=1.), name='epsilon_' + K.get_uid(''))(z_mean)
        #epsilon = Lambda(lambda x:K.random_normal(shape=(K.shape(x)[0], self.latent_dim), mean=0., stddev=1.), name='epsilon_' + str(K.get_uid('')))(z_mean)
        #z = Lambda(lambda x: x[0] + K.exp(0.5*x[1]) * x[2], name='z_' + K.get_uid(''))([z_mean, z_log_var, epsilon])
        z = self.sampling([z_mean, z_log_var])
        z_out = Lambda(lambda x: x, name='z')(z)
        encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
        #encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        
        d2 = Dense(32 * 32 * 128, activation='relu')(latent_inputs)
        r = Reshape((32, 32, 128))(d2)
        l4 = Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')(r)
        l5 = Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')(l4)
        l6 = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(l5)
        l7 = Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(l6)
        decoder = Model(latent_inputs, l7, name='decoder')
        return decoder

    def build_model(self):
        input_img = Input(shape=self.input_shape)
        z_mean, z_log_var, z = self.encoder(input_img)
        reconstruction = self.decoder(z)
        model = Model(input_img, reconstruction, name='rvae')
        reconstruction_loss = metrics.binary_crossentropy(input_img, reconstruction)
        reconstruction_loss *= self.input_shape[0] * self.input_shape[1]
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        model.add_loss(vae_loss)
        model.compile(optimizer='adam')
        return model
    
    def train(self, x_train, batch_size=32, epochs=50):
        loss_history = []
        for epoch in range(epochs):
            # Train for one epoch
            loss = self.model.train_on_batch(x_train, None)
            loss_history.append(loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")
        return loss_history
        


import os
import numpy as np
from PIL import Image

# Define paths to your image dataset
data_dir = '/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/combine'
class_names = os.listdir(data_dir)
img_size = 512

# Load images and preprocess
images = []
for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.
    images.append(img)

# Convert image list to numpy array
images = np.array(images)

# Split data into training and validation sets
x_train = images
print(x_train.shape)
train_size = int(0.8 * x_train.shape[0])
x_train, x_val = x_train[:train_size], x_train[train_size:]

# Instantiate the RVAE model
input_shape = x_train.shape[1:]
latent_dim = 32
rvae = rVAE(input_shape=input_shape, latent_dim=latent_dim)

# Train the RVAE model
rvae.train(x_train, batch_size=32, epochs=50)

# Evaluate the performance of the trained model
loss = rvae.model.evaluate(x_val)

# Generate new images by sampling from the latent space
z_sample = np.random.normal(size=(10, latent_dim))
x_decoded = rvae.decoder.predict(z_sample)
# Define RVAE model
#rvae = rVAE(input_shape=(img_size, img_size, 3), latent_dim=32)

# Train RVAE model
#rvae.train(images, batch_size=32, epochs=50)
