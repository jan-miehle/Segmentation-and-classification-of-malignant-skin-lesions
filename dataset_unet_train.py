import monai
from pathlib import Path
import cv2
from monai.transforms.utility.array import AsChannelFirst, EnsureChannelFirst
from numpy.lib.type_check import imag
import torchvision.transforms as torch_transforms

import matplotlib.pyplot as plt
import torch
from monai.transforms import (Compose,  RandGaussianSharpen, SpatialPad,ScaleIntensity, ToTensor, Zoom, LoadImage,RandScaleCrop,OneOf, RandHistogramShift,RandCoarseDropout,AddChannel,RandFlip,RandZoom, RandRotate,EnsureType,NormalizeIntensity, Flip,  RandAdjustContrast, RandGaussianSmooth,RandBiasField,RandStdShiftIntensity,  GaussianSharpen,RandGaussianNoise, HistogramNormalize, RandGibbsNoise, RandKSpaceSpikeNoise, Rotate, compose)
from torchvision.transforms.transforms import Resize
from PIL import Image
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from random import randint
import monai.visualize.img2tensorboard
import monai.utils
"""
Dataset for training, test and evaluation set of umet
"""
class data_set_u_net(monai.data.ImageDataset):
    """
    img_dir=None
    gt_dir=None
    size=512
    channels=3
    selection="all": to slect folder "hairs" or "clean" witrh selection of PH2 data set
    spatial_transform=None : for web interface
    intensity_transform=None : for web interface

    """
    def __init__(self, img_dir=None,gt_dir=None,size=512, channels=3, selection="all", spatial_transform=None,  intensity_transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spatial_transform =spatial_transform
        self.intensity_transform = intensity_transform
        self.channels = channels
        self.size = size

        self.img_files =[]
        self.seg_files = []
        #load images:
        if img_dir != None:
            for child in Path(img_dir).iterdir():
                self.img_files.append(img_dir + "/" + child.name)
            self.img_files.sort()
        else:
            if selection =="hairs":
                image_dir = "PH2Dataset/PH2 Dataset images hairs"
                
            elif selection =="clean":
                image_dir = "PH2Dataset/PH2 Dataset images no hairs"
            else:
                image_dir = "PH2Dataset/PH2 Dataset images"
            print("image dir: ", image_dir)
            for child in Path(image_dir).iterdir():
            #img = image_path +"/" + child.name
                img = image_dir + "/"+  child.name + "/" +child.name + "_Dermoscopic_Image" + "/" + child.name +".bmp"
                self.img_files.append(img)
            self.img_files.sort()

        #load gt files:
        if gt_dir != None:
            for child in Path(gt_dir).iterdir():
                self.seg_files.append(gt_dir + "/" + child.name)
            self.seg_files.sort()
        else:
            if selection =="hairs":
                gt_dir = "PH2Dataset/PH2 Dataset images hairs"
                
            elif selection =="clean":
                gt_dir = "PH2Dataset/PH2 Dataset images no hairs"
            else:
                gt_dir = "PH2Dataset/PH2 Dataset images"
            for child in Path(gt_dir).iterdir():
                gt = gt_dir + "/"+  child.name + "/" +child.name + "_lesion" + "/" + child.name+ "_lesion" +".bmp"
                self.seg_files.append(gt)
            self.seg_files.sort()       

      
      
 

    def __len__(self):
        return len(self.img_files)


    """
    called by dataloader
    idx (int) index of image

    returns:
    image: augmented image
    gt: augmented gt
    image_file: file name
    orig_image: image without augmentation
    orig_gt: gt without augmentation

    """   
    def __getitem__(self, idx):

        image_file =self.img_files[idx]   
        gt_file = self.seg_files[idx] 

        #load image:
        if self.channels != 1:
            type_transform = Compose([
                LoadImage(image_only=True),
                AsChannelFirst(),
                ToTensor(),
                EnsureType(device=self.device),
            ])
        else:
            type_transform = Compose([
                LoadImage(image_only=True),
                AddChannel(),
                ToTensor(),
                EnsureType(device=self.device),
            ])

        image = type_transform(image_file)

        #load gt
        type_transform_gt = Compose([
             LoadImage(image_only=True),
             AddChannel(),           
             ToTensor(),
             Resize((image.size()[1],image.size()[2])),
             EnsureType(device=self.device),
        ])
        
        gt = type_transform_gt(gt_file)
        orig_image = image
        orig_gt = gt       



        #add mask channel to image for equal transforming both together
        image = torch.cat((image,gt), 0)
    ################     spatial transforms
        if self.spatial_transform==None:
            self.spatial_transform =Compose([
                RandFlip(prob = 0.5),            
                RandRotate(range_x=0.9, prob=0.5, padding_mode="zeros"),
                OneOf([            
                    RandScaleCrop(roi_scale=0.7, max_roi_scale=1),
                    RandZoom(min_zoom=0.9, max_zoom=1.2, prob=1),
                    Compose()
                    ], weights=[0.5,0.5, 0.25])
                ])           

        
        image = self.spatial_transform(image)


        # split mask and image channels
        last_channel =len(image) -1
        gt=image[last_channel]
        type_transform_gt2=Compose([AddChannel()])
        gt = type_transform_gt2(gt)
        image = image[:last_channel]

       

    ######## intensioty transforms:
        if self.intensity_transform== None:
            self.intensity_transform= Compose([
                OneOf([
                    RandAdjustContrast(prob=1),
                    RandGaussianSharpen(prob=1),
                    RandGaussianSmooth(prob=1),
                    Compose()],weights=[0.2, 0.2, 0.6, 1]),
                OneOf([
                    RandGaussianNoise(prob=1),
                    RandGibbsNoise(prob=1),                
                    RandBiasField(prob=1),
                    Compose()
                    ], weights =[1, 0.5, 0.5, 3]),
                OneOf([
                    RandStdShiftIntensity(factors=(5,10), prob=0.1),
                    RandHistogramShift(),
                    Compose()
                
                ], weights=[1,1,1]),


                RandCoarseDropout(holes=20, spatial_size=30, prob=0.1, fill_value=0)          


        ])
        image = self.intensity_transform(image)



        fit_network_transform = Compose([
                                ToTensor(),
                                Resize((self.size, self.size)),
                                ScaleIntensity(), 
                                EnsureType(device=self.device),])
        
        #resize and cast to tensor for all images:
        image = fit_network_transform(image)
        gt = fit_network_transform(gt)            
        orig_image = fit_network_transform(orig_image)
        orig_gt = fit_network_transform(orig_gt)      
       
        

        return image, gt, image_file, orig_image, orig_gt
