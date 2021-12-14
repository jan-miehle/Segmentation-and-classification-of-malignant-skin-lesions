import cv2
from monai.transforms.utility.array import ToTensor
import pandas
import monai.metrics
from monai.transforms import Compose, AddChannel, Resize
from pathlib import Path
from tqdm import tqdm as tqdm
# metrics names to index pandas.DataFrame
metric_names= ["dice","hausdorff", "surface distance", "mask file", "gt_file"]


"""
compares two images with Dice Index, Hausdorff Distance and Surface Distance
"""
def compare_images(seg_file, gt_file):
    trans = Compose([
        AddChannel(),       
        Resize([512,512]),
        AddChannel(),
        ToTensor(),      
        ]
    )
    #read and rescale images
    #convert ot Tensor/ channel first image
    seg = cv2.imread(seg_file, 0) /255
    seg = trans(seg) 

    gt = cv2.imread(gt_file, 0) /255
    gt = trans(gt)

    #compute metrics
    dice = monai.metrics.compute_meandice(seg, gt)
    hausdorff = monai.metrics.compute_hausdorff_distance(seg, gt) 
    surface_distance = monai.metrics.compute_average_surface_distance(seg, gt) 

    #add to pands.Series
    m = pandas.Series([dice[0][0].item(), hausdorff[0][0].item(), surface_distance[0][0].item(), seg_file, gt_file], index=metric_names)
    return m


"""
calc metrics for two folders
metrics: Dice Index, Hausdorff Distance and Surface Distance

"""
def start_metrics(seg_dir, gt_dir):
    mask_files = []
    gt_files = []
    #read files:
    for child in Path(seg_dir).iterdir():
        m_file = seg_dir + "/" + child.name
        mask_files.append(m_file)
        filename = child.name.split(".")[0]
        gt_file = gt_dir +"/"+ filename + "_Segmentation.png"
        gt_files.append(gt_file)

    #collext metrics in series:
    list_of_series = [] 
    for mask, gt in tqdm(zip(mask_files, gt_files)):
        metric = compare_images(mask, gt)
        list_of_series.append(metric)
    #create pandas.DataFrame:
    data= pandas.DataFrame(list_of_series, columns=metric_names )
    data = data.sort_values('hausdorff')
    print(data)
    print(data.describe())
