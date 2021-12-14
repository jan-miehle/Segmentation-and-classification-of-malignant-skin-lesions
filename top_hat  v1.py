
import cv2
import numpy
import skimage.morphology
from pathlib import Path
import os
import tqdm

"""
runs the Top Hat Transformation on the expected PH2-data-set folder
small_elements (bool): which structuring element: True (5x5), False (7x7), None (diamond shape radius =3)
gauss  (bool): use guass filter
treshold  (bool):use thresholding
dilation (bool):  use dilation and erosion
save_path="ph2_": prefeix for generated save path
"""
def run_on_ph2(small_elements,gauss, treshold, dilation, save_path="ph2_"):
    save_path=save_path +"top_hat_v1/elemt="+str(small_elements)+"gauss_"+str(gauss)+"_treshold="+str(treshold)+"_dilation="+str(dilation)
    print(save_path)
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
    masks = tophat(images,small_elements, gauss, treshold, dilation)
    #inpaint and saving
    inpaint(images, masks, names, save_path)

"""
run Top-Hat Transform on specific image folder
small_elements (bool): which structuring element: True (5x5), False (7x7), None (diamond shape radius =3)
gauss  (bool): use guass filter
treshold  (bool):use thresholding
dilation (bool):  use dilation and erosion
save_path="ph2_": prefeix for generated save path
"""
def run(image_dir, small_elements,gauss, treshold, dilation,  save_path):
    save_path=save_path +"top_hat_v1/elemt="+str(small_elements)+"gauss_"+str(gauss)+" treshold="+str(treshold)+"_dilation="+str(dilation)

    images = []
    names =[]
    #load images:
    for child in Path(image_dir).iterdir():
        img = image_dir + "/"+  child.name 
        print(img)
        images.append(img)
        names.append(child.name.split(".")[0])
    #run top hat
    masks = tophat(images,small_elements, gauss, treshold, dilation)
    #inpaint and saving:
    inpaint(images, masks, names, save_path)


"""
calcualte pca for
img: input image numpy.array
"""
def pca(img):
    #store image shape for reshaping
    lenA =img.shape[0]
    lenB = img.shape[1]    
    #split channels:
    r = [[img[i][j][0] for j in range(lenB)] for i in range(lenA)]
    r = numpy.reshape(r, lenA*lenB)
    g = [[img[i][j][1] for j in range(lenB)] for i in range(lenA)]
    g= numpy.reshape(g, lenA*lenB)
    b = [[img[i][j][2] for j in range(lenB)] for i in range(lenA)]
    #reshape
    b= numpy.reshape(b, lenA*lenB)
    img = numpy.array([r, g, b,])

    # PCA
    mean, _, _ = cv2.PCACompute2(img, mean=None, maxComponents=1)

    #reshape
    img = numpy.reshape(mean , (lenA, lenB))  
    return img


def tophat(images, small_elements,gauss, treshold, dilation ):
    masks = []
    for img_path in tqdm.tqdm(images):
        img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        #calc pca is not singe channel
        if(isinstance(img[0][0],(numpy.uint8)) == False):
            img = pca(img)     

     
        structurng_elements = []
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
        else:
            f = skimage.morphology.diamond(5)
            structurng_elements.append(f) 
        mask = numpy.zeros((512,512), dtype=int)  
        ###black top hat with all structuring elements:
        for fp in structurng_elements:         
            m = skimage.morphology.black_tophat(img, fp) #das scheint ganz gut für haare zu sein footprint = skimage.morphology.diamond(3)
            mask = mask + m
        #rescaling
        mask = ( (mask/mask.max())*255 ).astype(numpy.uint8)    
        #cleaning:
        if gauss:
            mask = cv2.GaussianBlur(mask,(7,1),0)
            mask = cv2.GaussianBlur(mask,(1,7),0)
        
        if (treshold):
            mask[:][mask < 20] = 0  #+ skimage.morphology.black_tophat(img, skimage.morphology.rectangle(5, 1))

        if dilation:
            kernel = numpy.ones((3,3))

            mask = skimage.morphology.erosion(mask, kernel)
            mask = skimage.morphology.dilation(mask, kernel)
            mask = skimage.morphology.dilation(mask, kernel)     
        
   

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
    os.makedirs(mask_path, exist_ok=True)


    os.makedirs(save_path, exist_ok=True)
    print("start inpaint")
    for i in tqdm.tqdm(range(len(images))): 
        img = cv2.imread(images[i],cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))

        
        new_image  = img
        mask = masks[i].astype(numpy.uint8)
    
        new_image = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA) 
        
        file_name = names[i]+".png"
        new_file = cv2.imwrite(os.path.join(save_path , file_name),  numpy.array(new_image).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #save mask file:
        mask_file = cv2.imwrite(os.path.join(mask_path , file_name),  numpy.array(mask).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

       
run_on_ph2(False, True, True, True)
#run_on_ph2(False, True, True, False)
#run_on_ph2(False, False, True, True)
#run_on_ph2(False, False, True, False)

