import monai 
import cv2
import pandas as pd
import numpy
import torch
from monai.transforms import (Compose, AsChannelFirst,Resize,ScaleIntensity, RandGaussianSharpen,  ToTensor,  RandScaleCrop,OneOf, RandHistogramShift,RandCoarseDropout,AddChannel,RandFlip,RandZoom, RandRotate,EnsureType,   RandAdjustContrast, RandGaussianSmooth,RandStdShiftIntensity,  RandGaussianNoise,  RandGibbsNoise)




"""
Stores image data for training set for classifier network
does no augmentation
img_files: list if file names
labels: .csv
mask_files=None : list of file_names 
size=512 :image size
use_noise=True : If True, use intensity augmentation transform
return image, label, img_file

"""
class train_data(monai.data.Dataset):
    def __init__(self, img_files,labels, mask_files=None, size=512, use_noise=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_noise = use_noise
        self.img_labels = pd.read_csv(labels)
        self.img_files = img_files
        self.mask_files = mask_files
        self.size=size


        
        #transform for intensity augmentation
        self.trans_first_aug = Compose([
            ToTensor(),            
            RandCoarseDropout(holes=20, spatial_size=30, prob=0.1, fill_value=0), 
            OneOf([
                    RandAdjustContrast(prob=0.2,  gamma=(0.95, 1.05)),
                    RandGaussianSharpen(prob=0.2),
                    RandGaussianSmooth(prob=0.2),
                   
                   ],weights=[0.2, 0.2, 0.6]),
            OneOf([
                RandGaussianNoise(prob=0.75),
                RandGibbsNoise(prob=0.25),                
            ], weights =[0.5, 0.5]),
                
            OneOf([
                    RandStdShiftIntensity(factors=(0.8,1.0), prob=0.3),
                    RandHistogramShift(prob=0.3),                              
                ], weights=[1,1]),

            ToTensor()
        ])

        #transform for spatial augmentation
        self.trans_sec_aug = Compose([
                RandFlip(prob = 0.75),
                RandRotate(range_x=0.9, prob=0.8, padding_mode="zeros"),            
                OneOf([            
                    RandScaleCrop(roi_scale=0.7, max_roi_scale=0.99),
                    RandZoom(min_zoom=0.9, max_zoom=1.2, prob=0.07, keep_size=True),                
                ], weights=[0.1, 0.9]),

        ])
                
        #tranform for resizing and intensity scale:
        self.fit_network_transform = Compose([
                                ToTensor(),
                                Resize((self.size, self.size)),
                                ScaleIntensity(), 
                                EnsureType(device=self.device),])

    def __len__(self):
        return len(self.img_files) 


    """
    called by dataloader
    idx (int) index of image
    """   
    def __getitem__(self, idx):

        #load images_
        self.trans_load_image = Compose([
             #LoadImage(image_only=True),
             AsChannelFirst(),
             ToTensor(),
             Resize((self.size, self.size)),
             EnsureType(device=self.device),
             ])
        img_file_name =self.img_files[idx]  
        #
        image = cv2.cvtColor(cv2.imread(img_file_name, 1), cv2.COLOR_BGR2RGB)
        image = self.trans_load_image(image).to(self.device)

       
        #intensity augmentation
        if self.use_noise:
            image =self.trans_first_aug(image).to(self.device)
        image  = image.type(torch.int)
        # masking:
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
            # add mask as extra channel
            image = torch.cat((image,mask), 0)


        #spatial augmentation
        image =self.trans_sec_aug(image)

      
        #masking:
        if self.mask_files != None:
            #detach image and msk channels
            last_channel =len(image) -1
            mask=image[last_channel]
            type_transform_gt2=Compose([AddChannel(), ToTensor()])
            mask = type_transform_gt2(mask).int()           
            image = torch.tensor(image[:last_channel]).int()
            #mask images:           
            image = torch.bitwise_and(image,mask)           
           
            

        image = self.fit_network_transform(image) 
        #load labels:
        label = self.img_labels.iloc[idx]['label']
        if label == "benign":
            label = 0.0
        elif label =="malignant":
            label = 1.0
        else:
            label = float(label)
        label = numpy.array([label])      
       

        return image, label
