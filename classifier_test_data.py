import monai 
import cv2
from monai.transforms.utility.array import AsChannelLast
import pandas as pd
import numpy
import torch
from monai.transforms import (Compose, AsChannelFirst,Resize,ScaleIntensity, RandGaussianSharpen, SpatialPad, ToTensor, Zoom, LoadImage,RandScaleCrop,OneOf, RandHistogramShift,RandCoarseDropout,AddChannel,RandFlip,RandZoom, RandRotate,EnsureType,NormalizeIntensity, Flip,  RandAdjustContrast, RandGaussianSmooth,RandBiasField,RandStdShiftIntensity,  GaussianSharpen,RandGaussianNoise, HistogramNormalize, RandGibbsNoise, RandKSpaceSpikeNoise, Rotate, compose)


"""
Stores image data for test set for classifier network
does no augmentation
img_files: list of file names
labels: .csv
mask_files=None : list of file_names 
size=512 :image size

return image, label, img_file

"""
class test_data(monai.data.Dataset):
    def __init__(self, img_files,labels, mask_files=None, size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_labels = pd.read_csv(labels)
        self.img_files = img_files
        self.mask_files = mask_files
        self.size=size  
        
       
    def __len__(self):
        return len(self.img_files) 

    """
    called by dataloader
    idx (int) index of image
    """    
    def __getitem__(self, idx):
        #load image transform
        self.trans_load_image = Compose([
             AsChannelFirst(),
             ToTensor(),
             Resize((self.size, self.size)),
             EnsureType(device=self.device),
             ])
        #read image:
        img_file =self.img_files[idx]          
        image = cv2.cvtColor(cv2.imread(img_file, 1), cv2.COLOR_BGR2RGB)
        image = self.trans_load_image(image).to(self.device)  
        image  = image.type(torch.int)

        # if masking:
        if self.mask_files != None:
            #load mask
            mask_path = self.mask_files[idx]
            mask = cv2.imread(mask_path, 0)
            self.trans_load_mask = Compose([
                AddChannel(),           
                ToTensor(),
                Resize((self.size, self.size)),
                EnsureType(device=self.device),
             ])
            mask = self.trans_load_mask(mask).to(self.device)
            #mask image
            image = torch.bitwise_and(image,mask)  


        ##rescaling:
        self.fit_network_transform = Compose([
                                ToTensor(),
                                Resize((self.size, self.size)),
                                ScaleIntensity(), 
                                EnsureType(device=self.device),])
        

        image = self.fit_network_transform(image) 
        #load label:
        label = self.img_labels.iloc[idx]['label']
        if label == "benign":
            label = 0.0
        elif label =="malignant":
            label = 1.0
        else:
            label = float(label)
        label = numpy.array([label])    
       

        return image, label, img_file
