import os
import torch
import torchvision
from torchvision.datasets import ImageFolder

# Set directories for each folder of images
data_dirs = '/Users/kantg/OneDrive/Desktop/CMU/BMS Capstone/trial_pro_images/'

# Define a function to transform the images
transform = torchvision.transforms.Compose([
    #torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
])

# Use the ImageFolder dataset class to load the images
dataset = ImageFolder(root=data_dirs, transform=transform)

# Use a data loader to create batches of images
batch_size = 200
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over the batches and print the shape of each batch
for batch,image in data_loader:
    print(batch.shape)
    
print(len(data_loader.dataset))