from pathlib import Path
import numpy
import cv2
import os
import tqdm as tqdm
from matplotlib import pyplot as plt

"""
combines a folder of 3-channel-images and a folder of 1-channel-images to a set of 4-channel-images 
source_1: folder of 3-channel images. If None, the expected folder-structur for PH^2 data set will be read
source_2: folder for 1-channel images
out_path: save_path for resulting images 

"""
def combine (source_1, source_2, out_path):
    ##to read images:
    images1 = []
    images2 = []
    ##read source_1 or ph2 data set
    if source_1 != None:
        for child in Path(source_1).iterdir():
            images1.append(source_1 + "/" + child.name)
    else: 
        image_dir = "PH2Dataset/PH2 Dataset images"
        for child in Path(image_dir).iterdir():
            img = image_dir + "/"+  child.name + "/" +child.name + "_Dermoscopic_Image" + "/" + child.name +".bmp"
            print(img)
            images1.append(img)
    #read source_2
    for child in Path(source_2).iterdir():
        images2.append(source_2 + "/" + child.name)
    images1.sort()
    images2.sort()

    os.makedirs(out_path, exist_ok=True)
    #combine all images:
    for i in range(len(images1)):
        #read and resize:
        x = cv2.cvtColor(cv2.imread(images1[i], 1), cv2.COLOR_BGR2RGB)        
        x = cv2.resize(x, (512,512))
        y = cv2.imread(images2[i], 0)
        y = cv2.resize(y, (512,512))
        
        #combine image channels:
        r, g, b = cv2.split(x)
        new_image = cv2.merge((r,g ,b ,y)) 
        #save:
        file_name = images1[i].split(".")[0]
        file_name= file_name.split("/")[len(file_name.split("/")) - 1]
        file_name = file_name +".png"
        print(file_name)
        test =cv2.imwrite(os.path.join(out_path , file_name),  numpy.array(new_image).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #prints True if sucessfully write file
        print(test)

#combine("ISIC/ISBI2016_ISIC_Part3B_Test_Data", "ISIC/test_set/lc/lbp", "ISIC/test_set/combined")