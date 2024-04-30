# Loading librarys

import os
import zipfile
import numpy as np
import pandas as pd

import ast
import cv2

import torch

from sklearn.model_selection import train_test_split
import shutil
from tqdm.notebook import tqdm
import tqdm.notebook as tq

import albumentations as albu
from albumentations import Compose

import matplotlib.pyplot as plt


#**********************************************

from IPython.display import Image, clear_output

clear_output()
print(f"Setup complete. Using torch {torch.version} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

#**********************************************************

!git clone https://github.com/ultralytics/yolov5.git  

#change the working directory to yolov5
%cd yolov5
os.chdir('/kaggle/working/yolov5')

#install dependencies
%pip install -r requirements.txt 

#Change the working directory back to /kaggle/working/
os.chdir('/kaggle/working/')

!pwd

#***********************************************************
import shutil
shutil.make_archive("Dataset", 'zip', "/kaggle/input/final-leave-dataset")
#***********************************************************

!unzip -q Dataset.zip -d /kaggle/working/datasets && rm Datasetmp.zip

#***********************************************************

def draw_bbox(image, xmin, ymin, xmax, ymax, text=None):
    
    """
    This functions draws one bounding box on an image.
    
    Input: Image (numpy array)
    Output: Image with the bounding box drawn in. (numpy array)
    
    If there are multiple bounding boxes to draw then simply
    run this function multiple times on the same image.
    
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a 
    title above the bbox.
    
    xmin, ymin --> coords of the top left corner.
    xmax, ymax --> coords of the bottom right corner.
    
    """


    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................
    
    start_point = (xmin, ymin) 
    end_point = (xmax, ymax) 
    bbox_color = (255, 0, 0) 
    bbox_thickness = 15

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness) 
    
    
    
    # Draw the tbackground behind the text and the text
    # .................................................
    
    # Only do this if text is not None.
    if text:
        
        # Draw the background behind the text
        text_bground_color = (0,0,0) # black
        cv2.rectangle(image, (xmin, ymin-150), (xmin+w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255) # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin-30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font, 
                           fontScale, text_color, thickness, cv2.LINE_AA)



    return image
#**********************************************************************


trainimg = []
valimg = []

for im in os.listdir("/kaggle/working/datasets/Data/images/train"):
    trainimg.append(im)

for im in os.listdir("/kaggle/working/datasets/Data/images/val"):
    valimg.append(im)

#**********************************************************************

import random

def display_images():

    # set up the canvas for the subplots
    plt.figure(figsize=(20,70))


    for i in range(1,13):

        index = i
        
        num = random.randint(0,len(trainimg))
        # Load an image
        path = "/kaggle/working/datasets/Data/images/train/" + trainimg[num] 
        image = plt.imread(path)
        #image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        plt.subplot(10,3,i)

        plt.imshow(image)
        plt.axis('off')
#***********************************************************************

text_file_list = os.listdir('/kaggle/working/datasets/Data/labels/train')

text_file = text_file_list[0]

#***********************************************************************

# Display the contents of a text file
! cat '/kaggle/input/final-leave-dataset/Data/labels/train/Image_27_y10199.txt'
#***********************************************************************

# Create the yaml file called my_data.yaml and We will save this file inside the yolov5 folder.

yaml_dict = {'train': '/kaggle/working/datasets/Data/images/train',   # path to the train folder
            'val': '/kaggle/working/datasets/Data/images/val', # path to the val folder
            'nc': 18,                             # number of classes
            'names': ['AJ','AT','CR','CK','w','DT','DD','ES','CIK','IL','ML','NS','N','OE','PM',str(999),'PR','VN']}                # list of label names
#***********************************************************************
import yaml

with open(r'/kaggle/working/yolov5/my_data.yaml', 'w') as file:
    documents = yaml.dump(yaml_dict, file)
#************************************************************************
!cat '/kaggle/working/yolov5/my_data.yaml'
#************************************************************************
! WANDB_MODE="dryrun" python train.py --img 640 --batch 16 --epochs 200 --data /kaggle/working/yolov5/my_data.yaml --weights /kaggle/working/yolov5/yolov5s.pt
#************************************************************************
os.listdir("/kaggle/working/yolov5/runs/train/exp4")
#************************************************************************
plt.figure(figsize = (15, 15))
plt.imshow(plt.imread('/kaggle/working/yolov5/runs/train/exp4/train_batch0.jpg'))
plt.show()
