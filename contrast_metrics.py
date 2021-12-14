import numpy
import cv2
import pandas
import skimage.measure    
from matplotlib import pyplot as plt
from pathlib import Path


"""
compares some metrics for a list of image files
image_files: list of files to meassure
plot_hist (bool): should the histrogram of the images be plottet?
returns: pandas.dataframe with metric calues
"""
def eval(image_files, plot_hist):
    images = []
    #read images and resize_
    for f in image_files:
        i = cv2.cvtColor(cv2.imread(f, 1), cv2.COLOR_BGR2RGB)
        i = cv2.resize(i,[512,512])
        images.append((i,f))
    #stores list with metrics for each image:       
    p = []
    for (image, file) in images:    

        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  
        #calc image blur:
        blur= skimage.measure.blur_effect(grey_image) 


        #RMS contrast:
        normalized = grey_image / 255
        rms_contrast = normalized.std()

        ####michelson contrasat
        min = int(numpy.min(grey_image))
        max = int(numpy.max(grey_image))    
        michelson_contrast = round((max-min)/(max+min), 3)

        #count number of areas:
        #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
        _ , number_of_areas = skimage.measure.label(grey_image, return_num=True)        

        #save:
        p.append([round(blur, 3), round(rms_contrast,3), round(michelson_contrast,3), round(number_of_areas,3), file.split('/')[len(file.split('/')) - 1]]) 
    #build pandas.DataFrame with data from all images: 
    panda = pandas.DataFrame(p,columns=['blur', 'rms contrast', 'michelson contrast',  'number_of_areas', 'file name'])  #'histogram entropy',
    print(panda)
    print(panda.describe())
    return panda



"""
reads a folder with images and calls eval() for all images in it
folder (str): image folder
plot_hist (bool): should the histrogram of the images be plottet?
file_name=None : if filename is given, the results will be saved in file_name.csv
"""
def eval_folder(folder, plot_hist, file_name=None):
    files =[]
    for child in Path(folder).iterdir():
        file = folder +"/" +child.name
        files.append(file)
    files.sort()
    panda = eval(files, plot_hist)
    if file_name is not None:
        file_name =file_name + '.csv'
        panda.to_csv(file_name)


