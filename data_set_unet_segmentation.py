import monai
from pathlib import Path
from monai.transforms.intensity.array import ScaleIntensity
from monai.transforms.utility.array import AsChannelFirst

import torch
from monai.transforms import (Compose,    ToTensor,  LoadImage, AddChannel, EnsureType, )
from torchvision.transforms.transforms import Resize
import monai.visualize.img2tensorboard
import monai.utils

"""
Dataset for segmenting images with pretrained u-net


"""
class data_set_segmentation(monai.data.ImageDataset):
    """
    img_dir=None
    size=512
    channels=3
    selection="all": to select folder "hairs" or "clean" with selection of PH2 data set


    """
    def __init__(self, img_dir=None,size=512, channels=3, selection="all"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = channels
        self.size = size

        self.img_files =[]
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
                img = image_dir + "/"+  child.name + "/" +child.name + "_Dermoscopic_Image" + "/" + child.name +".bmp"
                self.img_files.append(img)
            self.img_files.sort()            
 

    def __len__(self):
        return len(self.img_files)
    
    """
    called by dataloader
    idx (int) index of image

    returns:
    image: augmented image
    image_file: file name

    """   
    def __getitem__(self, idx):
        image_file =self.img_files[idx]   
        #load images:
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

        # to tensor, scale image size
        fit_network_transform = Compose([
                                ToTensor(),
                                Resize((self.size, self.size)),
                                ScaleIntensity(), 
                                EnsureType(device=self.device),])
        image = fit_network_transform(image)     
       
        
        return image, image_file
