from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np


# Define the input shape
input_shape = (512*512*3,)

# Define the size of the latent space
latent_dim = 32

# Define the number of neurons in each layer of the encoder and decoder
encoder_layer_sizes = [512, 256, 128]
decoder_layer_sizes = [128, 256, 512]

# Define the standard deviation for the normal distribution used for sampling
epsilon_std = 1.0

# Define the rotation-invariant layer
def rotation_invariant_layer(x):
    x = K.reshape(x, (-1, 28, 28))
    x = K.permute_dimensions(x, (0, 2, 1))
    x = K.reshape(x, (-1, 28 * 28))
    return x

# Define the encoder
def build_encoder():
    # Define the input layer
    input_img = Input(shape=input_shape, name='encoder_input')

    # Apply the rotation-invariant layer
    x = Lambda(rotation_invariant_layer, name='rotation_invariant')(input_img)

    # Add the hidden layers
    for layer_size in encoder_layer_sizes:
        x = Dense(layer_size, activation='relu')(x)

    # Compute the mean and log variance of the latent space
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Use the mean and log variance to sample from the latent space
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # Sample from the latent space
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Define the encoder model
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')

    return encoder

# Define the decoder
def build_decoder():
    # Define the input layer
    input_z = Input(shape=(latent_dim,), name='decoder_input')

    # Add the hidden layers
    x = input_z
    for layer_size in decoder_layer_sizes:
        x = Dense(layer_size, activation='relu')(x)

    # Compute the output of the decoder
    output_img = Dense(input_shape[0], activation='sigmoid', name='decoder_output')(x)

    # Define the decoder model
    decoder = Model(input_z, output_img, name='decoder')

    return decoder
    

class rVAE:
    def __init__(self, input_dim, latent_dim, enc_layers=[128, 64], dec_layers=[64, 128], act_fn='relu'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.act_fn = act_fn
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()
        
    def build_encoder(self):
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.enc_layers:
            x = Dense(units, activation=self.act_fn)(x)
        
        # Output layers
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs
        
        # Hidden layers
        for units in self.dec_layers:
            x = Dense(units, activation=self.act_fn)(x)
        
        # Output layer
        outputs = Dense(self.input_dim, activation='sigmoid')(x)
        
        return Model(latent_inputs, outputs, name='decoder')
    
    def build_vae(self):
        inputs = self.encoder.inputs[0]
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Define VAE loss
        reconstruction_loss = mse(inputs, reconstructed)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        
        # Define custom layer for rotational invariance
        ri_layer = Lambda(self.rotational_invariance, output_shape=(self.latent_dim,), name='rotational_invariance')
        z_ri = ri_layer(z)
        
        # Combine loss and RI layer into one model
        vae_ri = Model(inputs, [reconstructed, z_ri])
        vae_ri.add_loss(vae_loss)
        vae_ri.compile(optimizer='adam')
        
        return vae_ri
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def rotational_invariance(self, z):
        # Compute the mean of the circular coordinates
        r = K.sqrt(K.sum(K.square(z), axis=-1, keepdims=True))
        theta = K.atan2(z[:, 1], z[:, 0])
        mean_theta = K.mean(theta)
        
        # Subtract mean and recompute coordinates
        theta = theta - mean_theta
        x = r * K.cos(theta)
        y = r * K.sin(theta)
        
        return Concatenate()([x, y])


# training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the input shape based on your image dimensions and channels
input_shape = (512, 512, 3)

# Create an instance of the rVAE model
rvae = rVAE(input_dim=input_shape, latent_dim=32)

# Compile the model
rvae.compile(optimizer='adam')

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/combine',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='input')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    '/jet/home/thanhngp/BMS1/ASO_Xstals_FInal_Time_Point/combine',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='input')

# Train the model
rvae.fit(train_generator,
        epochs=50,
        validation_data=valid_generator)
        
