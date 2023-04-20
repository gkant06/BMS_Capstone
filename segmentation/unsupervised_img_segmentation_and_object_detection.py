# This pytorch code generates segmentation labels of an input image - unsupervised segmentation. Further, segmented images are used for object detection and generating a metadata file with information for each object.Each object in an image is cropped and stored in a folder.
# Need to input a file with reference images (ref) and test images (test)
# No. of reference images (ref) > = 1
# No. of test images (test) > = 1
# Reference: https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
# Reference: Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering
# Object detection: Using Otsu Thresholding and generation bounding boxes


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
from datetime import datetime
import tqdm
import csv

## User Input  - directory location which contains ref and test image folders
image_dir = '/jet/home/gkant/sample3/'

use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--maxUpdate', metavar='T', default=1000, type=int, 
                    help='number of maximum update count')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--batch_size', metavar='bsz', default=1, type=int, 
                    help='number of batch_size')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FOLDERNAME',
                    help='input image folder name',required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float, 
                    help='step size for continuity loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

img_list = sorted(glob.glob(args.input+'ref/*'))
#img_list = sorted(glob.glob(args.input+'*'))
print(img_list)
im = cv2.imread(img_list[0])

# train
model = MyNet( im.shape[2] )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(args.maxIter):
    print('Training started. ' + str(batch_idx+1)+' / '+str(args.maxIter))
    for im_file in range(int(len(img_list)/args.batch_size)):
        for loop in tqdm.tqdm(range(args.maxUpdate)):
            im = []
            for batch_count in range(args.batch_size):
                # load image
                resized_im = cv2.imread(img_list[args.batch_size*im_file + batch_count])
                resized_im = cv2.resize(resized_im, dsize=(224, 224))
                resized_im = resized_im.transpose( (2, 0, 1) ).astype('float32')/255.
                im.append(resized_im)

            data = torch.from_numpy( np.array(im) )
            if use_cuda:
                data = data.cuda()
            data = Variable(data)
    
            HPy_target = torch.zeros(data.shape[0], resized_im.shape[1]-1, resized_im.shape[2], args.nChannel)
            HPz_target = torch.zeros(data.shape[0], resized_im.shape[1], resized_im.shape[2]-1, args.nChannel)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            # forwarding
            optimizer.zero_grad()
            output = model( data )
            output = output.permute( 0, 2, 3, 1 ).contiguous().view( data.shape[0], -1, args.nChannel )

            outputHP = output.reshape( (data.shape[0], resized_im.shape[1], resized_im.shape[2], args.nChannel) )
    
            HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
            HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]    
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            output = output.reshape( output.shape[0] * output.shape[1], -1 )
            ignore, target = torch.max( output, 1 )

            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
            loss.backward()
            optimizer.step()

    #torch.save(model.state_dict(), os.path.join(args.input, 'b'+str(args.batch_size)+'_itr'+str(args.maxIter)+'_layer'+str(args.nConv+1)+'.pth'))
    torch.save(model, os.path.join(args.input, 'b'+str(args.batch_size)+'_itr'+str(args.maxIter)+'_layer'+str(args.nConv+1)+'.pth'))

label_colours = np.random.randint(255,size=(100,3))
test_img_list = sorted(glob.glob(args.input+'test/*'))
if not os.path.exists(os.path.join(args.input, 'result/')):
    os.mkdir(os.path.join(args.input, 'result/'))
print('Testing '+str(len(test_img_list))+' images.')
for img_file in tqdm.tqdm(test_img_list):
    im = cv2.imread(img_file)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    inds = target.data.cpu().numpy().reshape( (im.shape[0], im.shape[1]) )
    inds_rgb = np.array([label_colours[ c % args.nChannel ] for c in inds])
    inds_rgb = inds_rgb.reshape( im.shape ).astype( np.uint8 )
    #print(inds_rgb)
    print(inds_rgb.shape)
    cv2.imwrite( os.path.join(args.input, 'result/') + os.path.basename(img_file), inds_rgb )
    
    
## Objects are cropped from the original image

# Define input and output directories

input_dir = image_dir + 'result/'
original_dir = image_dir + 'test/'
output_dir = image_dir + 'output_images/'
metadata_file = image_dir + 'metadata.csv'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize metadata list
metadata = []

# Process each image in input directory
for filename in os.listdir(input_dir):
    # Check if file is an image"
    if not filename.endswith(('png', 'jpg', 'jpeg')):
        continue
    
    # Load input and original image, grayscale, Otsu's threshold
    input_image = cv2.imread(os.path.join(input_dir, filename))
    original_image = cv2.imread(os.path.join(original_dir, filename))
    if input_image is None or original_image is None:
        print(f"Error: Could not load image {filename}")
        continue

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # invert the image such that blobs (which take lesser area in image are 255 (black))
    if np.where(opening == 255)[0].shape[0] > np.where(opening == 0)[0].shape[0]:
        opening = 255 - opening
    
    # Find contours, obtain bounding box, extract and save ROI
    object_number = 0
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (36,255,12), 2)
        objects = original_image[y:y+h, x:x+w]
        output_filename = os.path.join(output_dir, 'object_{}_{}.png'.format(os.path.splitext(filename)[0], object_number))
        cv2.imwrite(output_filename, objects)
        object_number += 1

        # Append metadata for current ROI
        metadata.append([output_filename, filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (x, y, w, h)])

# Create metadata CSV file
with open(metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'parent_image', 'timestamp', 'object_location', 'object_count_in_parent_img'])
    
    # Count child images for each parent image and write metadata to CSV file
    for filename in os.listdir(input_dir):
        parent_image = filename
        child_image_count = 0
        for metadata_item in metadata:
            if metadata_item[1] == parent_image:
                writer.writerow([metadata_item[0], parent_image, metadata_item[2], metadata_item[3], child_image_count])
                child_image_count += 1
                
print('Process Complete')
                