
import scipy
from sklearn.cluster import k_means as k_means
from skimage import feature
from pathlib import Path
import cv2
import numpy
from skimage.color import rgb2lab
import tqdm as tqdm
import time
import cv2
import os



"""
calcualte LC-Algorithm for image
implementation of
Pedro M.M. Pereira, Rui Fonseca-Pinto, Rui Pedro Paiva, Pedro A.A. Assuncao, Luis M.N. Tavora, Lucas A. Thomaz, Sergio M.M. Faria,
Dermoscopic skin lesion image segmentation based on Local Binary Pattern Clustering: Comparative study,
Biomedical Signal Processing and Control,
Volume 59,
2020,
101924,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2020.101924.

and 
P. M. M. Pereira, R. Fonseca-Pinto, R. P. Paiva, L. M. N. Tavora, P. A. A. Assuncao and S. M. M. de Faria, 
"Accurate Segmentation of Dermoscopic Images based on Local Binary Pattern Clustering," 
2019 42nd International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), 2019, pp. 314-319, doi: 10.23919/MIPRO.2019.8757023.
imagePath: path of image


return lbp_ local binary pattering, lyl: connected image of LBP and Y, lab: numpy array of image in L*a*b color, mask: segmentation mask
"""
def lc(imagePath):
    imagePath = imagePath

    #to count runtime:
    startime =  time.perf_counter()


    numpyImage = cv2.cvtColor(cv2.imread(imagePath, 1), cv2.COLOR_BGR2RGB)     
    size = [numpyImage.shape[0], numpyImage.shape[1]]
    
    
    ###strat lbp processing:
    #save lbp for feature extraction!
    lbp, lbp2, l, y =lbpProcessing(numpyImage, size)

    ##Add the lbp processed image and luminance input //(+) - operation
    lyl = addImages(l, y, size)    

    # skimage.color  convert:
    lab = rgb2lab(lyl)

    #k-means++
    c = cluster(lab, size)
    mask = c.astype(bool)
  
    print("runtime: " + str(  time.perf_counter() - startime) + " sec")
    
    return lbp , lyl, lab, mask





      


 
def lbpProcessing(numpyImage, size):   
    z = numpy.array([[[numpyImage[i][j][0] * 0.2989,numpyImage[i][j][1] * 0.5870, numpyImage[i][j][2] * 0.114] for j in range(size[1])] for i in tqdm.tqdm(range(size[0]))])     
    y = numpy.zeros(size)
    # intensity image:
    y = numpy.array([[int(z[i][j][0]) +  int(z[i][j][1]) +  int(z[i][j][2])  for j in range(size[1])] for i in tqdm.tqdm(range(size[0]))])
    y = numpy.uint8(y) 

    #calculate lbp:
    lbp = feature.local_binary_pattern(y, 8, 1) 
    lbp2 = stepTwoLBP(lbp, size)
    l = gaussianFilterLBP(lbp2)
    return lbp, lbp2, l, y


def stepTwoLBP(lbp, size):
    print("calc pow 2")
    # only keep lbp group 0 and 1
    newMatrix = numpy.array([[calcPow2(lbp[i][j])  for j in range(size[1])] for i in tqdm.tqdm(range(size[0]))]) 
    return newMatrix 

"""
is patter n part of lbp group 0 or 1
"""
def calcPow2 (x):
    x = int(x)       
    if ((x == 2) or (x == 4) or (x == 8) or (x == 16) or (x == 32) or (x == 64) or (x == 128) or (x == 1) or (x == 0)): 
        return 0
                                    
    return 1

     

def gaussianFilterLBP(lbp2):
    sigma = 3
    windowSize = 13
    t =  (((windowSize - 1)/2)-0.5)/sigma
    l = scipy.ndimage.filters.gaussian_filter(lbp2 * 255, sigma, truncate=t) 
    l = numpy.array(l)    
    
    return l


"""Add LBP and Y to [L,Y,L]"""
def addImages(l, y, size):  
    print("building LYL:")
    lyl = numpy.array([[[l[i][j],y[i][j],l[i][j]] for j in range(size[1])] for i in tqdm.tqdm(range(size[0]))])    
    lyl = numpy.array(lyl)
    return lyl  

"""
start k-means++ algorithm
"""
def cluster(lab, size):
    x = numpy.reshape(lab, (lab.shape[0]*lab.shape[1] ,lab.shape[2]))

    _, label, _,  = k_means(x, n_clusters=2, init='k-means++', n_init=3, max_iter=100,)
    cluster = numpy.reshape(label, (size[0], size[1]))
    cluster = cluster.astype(bool)

    cluster = chooseCluster(cluster, lab)

    return cluster

"""
detect correct cluster
"""
def chooseCluster(c, lab):
    averageAtrue = 0
    averageAfalse = 0
    countTrue = 0
    countFalse = 0
    averageBtrue = 0
    averageBfalse = 0

    print("Choose correct cluster:")
    for i in tqdm.tqdm(range(len(lab))):
        for j in range(len(lab[i])):
            if (c[i][j] == True):
                countTrue +=1
                averageAtrue += lab[i][j][1]
                averageBtrue += lab[i][j][2]
            else:
                countFalse +=1
                averageAfalse += lab[i][j][1]
                averageBfalse += lab[i][j][2]
    """
    hier < / > vertauschen, um True/False-Verteilung zu tauschen
    """
    if (((averageAtrue + averageBtrue)/countTrue) < ((averageAfalse + averageBfalse)/countFalse)):
        c = numpy.invert(c)
        print("##########################            INVERT             ##########################################")
    return c





"""
run lc-algo on given path
"""   
def run_lc(image_path, save_path):
    #create folders:
    lbp_save_path =os.path.join(save_path , "lbp")
    os.makedirs(lbp_save_path, exist_ok=True)
    lyl_save_path = os.path.join(save_path ,"lyl")
    os.makedirs(lyl_save_path, exist_ok=True)
    mask_save_path = os.path.join(save_path , "mask")
    os.makedirs(mask_save_path, exist_ok=True)
    lap_save_path = os.path.join(save_path, "lab")
    os.makedirs(lap_save_path, exist_ok=True)


    for child in Path(image_path).iterdir():
        #img = image_path +"/" + child.name
        if (image_path =="PH2Dataset/PH2 Dataset images"):
            img = image_path + "/"+  child.name + "/" +child.name + "_Dermoscopic_Image" + "/" + child.name +".bmp"
        else:
            img = image_path + "/"+  child.name 
        print(img)
        lbp, lyl, lab, mask = lc(img)

        #save:
        file_name = child.name +".bmp"
        new_file_lbp = cv2.imwrite(os.path.join(lbp_save_path , file_name),  numpy.array(lbp).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(new_file_lbp)
        new_file_lyl = cv2.imwrite(os.path.join(lyl_save_path, file_name),  numpy.array(lyl).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(new_file_lyl)  
        new_file_mask = cv2.imwrite(os.path.join(mask_save_path,file_name),  numpy.array(mask * 255).astype(int),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(new_file_mask)
     
        numpy.savez_compressed(os.path.join(lap_save_path, child.name), lab)


