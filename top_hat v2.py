
import cv2
import numpy
from pathlib import Path
import os
import tqdm
import skimage.morphology

"""
runs the Top Hat Transformation on the expected PH2-data-set folder
small_elements (bool): which structuring element: True (5x5), False (7x7), None (diamond shape radius =3)
gauss  (bool): use guass filter
treshold  (bool):use thresholding
dilation (bool):  use dilation and erosion
save_path="ph2_": prefeix for generated save path
"""
def run_on_ph2(small_elements, gauss, treshold, dilation, save_path="ph2_"):
    save_path=save_path +"top_hat_v2/step3/small_elements="+str(small_elements)+"_gauss="+str(gauss)+"_treshold="+str(treshold)+"_dilation="+str(dilation)
    images = []
    names =[]
    #load images:
    image_dir = "PH2Dataset/PH2 Dataset images"
    for child in Path(image_dir).iterdir():
        img = image_dir + "/"+  child.name + "/" +child.name + "_Dermoscopic_Image" + "/" + child.name +".bmp"
        print(img)
        images.append(img)
        names.append(child.name)
    #run top hat
    masks = tophat(images, small_elements, gauss, treshold, dilation)
    #inpaint and saving:

    inpaint(images, masks, names, save_path)

"""
run Top-Hat Transform on specific image folder
small_elements (bool): which structuring element: True (5x5), False (7x7), None (diamond shape radius =3)
gauss  (bool): use guass filter
treshold  (bool):use thresholding
dilation (bool):  use dilation and erosion
save_path="ph2_": prefeix for generated save path
"""
def run(image_dir, small_elements, gauss, treshold, dilation, save_path):
    save_path=save_path +"/small_elements="+str(small_elements)+"_gauss="+str(gauss)+"_treshold="+str(treshold)+"_dilation="+str(dilation)
    images = []
    names =[]
    #load images:
    for child in Path(image_dir).iterdir():
        img = image_dir + "/"+  child.name 
        print(img)
        images.append(img)
        names.append(child.name.split(".")[0])
    #run top hat
    masks = tophat(images, small_elements, gauss, treshold, dilation)    
    #inpaint and saving:
    inpaint(images, masks, names, save_path)



def tophat(images, small_elements, gauss, treshold, dilation):
    masks = []
    print("calcualte top hat")
    for img_path in tqdm.tqdm(images):
        img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor( cv2.imread(img_path,cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512,512))
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except:
           pass
        structurng_elements = []
        ##########structuring elements version 1:        
        if small_elements==True:
            f = numpy.array([
                            [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [0,0,0,0,0],
                            [0,0,0,0,0],
                            [1,1,1,1,1],
                            [0,0,0,0,0],
                            [0,0,0,0,0]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [1,0,0,0,0],
                            [0,1,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,1,0],
                            [0,0,0,0,1]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [0,0,0,0,1],
                            [0,0,0,1,0],
                            [0,0,1,0,0],
                            [0,1,0,0,0],
                            [1,0,0,0,0]
            ])
            structurng_elements.append(f)
            f = skimage.morphology.diamond(5)
            structurng_elements.append(f)
        elif small_elements==False:
            #structuring elements version2
            
            f = numpy.array([
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,1,0,0,0]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0],
                            [1,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [1,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0],
                            [0,0,1,0,0,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,0,0,1,0,0],
                            [0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1]
            ])
            structurng_elements.append(f)
            f = numpy.array([
                            [0,0,0,0,0,0,1],
                            [0,0,0,0,0,1,0],
                            [0,0,0,0,1,0,0],
                            [0,0,0,1,0,0,0],
                            [0,0,1,0,0,0,0],
                            [0,1,0,0,0,0,0],
                            [1,0,0,0,0,0,0]
            ])
            structurng_elements.append(f)
            f = skimage.morphology.diamond(7)
            structurng_elements.append(f)
        ###singe structurering element:
        else:
            f = skimage.morphology.diamond(5)
            structurng_elements.append(f) 
        
        ## histogram equalisation
        contrast_image = cv2.equalizeHist(img)
        n = numpy.zeros((512,512), dtype=int)
        m = numpy.zeros((512,512), dtype=int)  
        ###black top hat for contrast_image and img with all structuring elements:
        for fp in structurng_elements:         
            mask = skimage.morphology.black_tophat(img, fp) #das scheint ganz gut für haare zu sein footprint = skimage.morphology.diamond(3)
            mask2 =  skimage.morphology.black_tophat(contrast_image, fp)
            n = n + mask
            m = m + mask2 
        #combine mask:
        mask = m +n
           
        #rescaling:
        mask = ( (mask/mask.max())*255 ).astype(numpy.uint8) 
        #cleaning:   
        if gauss:
            mask = cv2.GaussianBlur(mask,(7,1),0)
            mask = cv2.GaussianBlur(mask,(1,7),0)
        if treshold: 
            mask[:][mask < 20] = 0
        
        #################opening/closing:  
        if dilation:  
            kernel= numpy.ones((3,3),numpy.uint8)
            mask =cv2.erode(mask,kernel)
            mask =cv2.dilate(mask,kernel)
            mask =cv2.dilate(mask,kernel)
        
        masks.append(mask)        
        

    return masks

"""
inpaints images with masks
images: list of images
masks: list of maks
names: file name
save_path: save inpainted image with failename here

"""
def inpaint(images, masks, names, save_path):
    mask_path = save_path + "_masks"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    print("start inpaint")
    for i in tqdm.tqdm(range(len(images))): #), m in zip(images, masks):
        img = cv2.imread(images[i])
        img = cv2.resize(img, (512,512))


        
        new_image  = img
        mask = masks[i].astype(numpy.uint8)
    
        new_image = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)         
        file_name = names[i]+".png"
        new_file = cv2.imwrite(os.path.join(save_path , file_name),  numpy.array(new_image).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        mask_file = cv2.imwrite(os.path.join(mask_path , file_name),  numpy.array(mask).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
       





