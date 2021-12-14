from cv2 import rotate, transform
import streamlit as st
import monai
import monai.transforms

import pandas as pd
import sys
sys.path.append("..")
import state
from  unet import unet as unet
"""
displays page for training
"""
def show():
    epochs = st.number_input('Select trainign epochs', min_value=1, value=2)

    if st.button('Start training'):
        #build loss function
        if  state.loss_name =="Dice Loss":
                loss_fn = monai.losses.DiceLoss(include_background=True, jaccard= state.jac,  squared_pred= state.sq) 
        elif state.loss_name =="Dice CE Loss":
                loss_fn = monai.losses.DiceCELoss(include_background=True, jaccard= state.jac,  squared_pred= state.sq)
        else:
                loss_fn = monai.losses.DiceFocalLoss(include_background=True, jaccard= state.jac,  squared_pred= state.sq)
            
        x = state.smallest_layer
        mul = 1
        layers = []
        #create stride-size input:
        while x < state.largest_layer:
            x = state.smallest_layer * mul
            layers.append(x)       
            mul = mul *2
        stride_len =len(layers) -1
        strides = [2 for i in range(stride_len)]
        strides = tuple(strides)
        layers = tuple(layers)

        #build augmentation pipelines
        trans = build_transforms()
        int_trans = build_intensity_transform()
        #instantiate unet
        network = unet(img_dir=None,drop=state.dropout, save_path=None, network_name="web_train", size=state.size, channels=state.channels, layers=layers, strides=strides, loss_function=loss_fn,spatial_transform=trans,intensity_transform=int_trans, jac=state.jac,web=True)


        #state traing

        network.get_data(batch_size=4)
        x, losses, v_losses = network.train(epochs=epochs)
        df = pd.DataFrame(
        {
            'train loss': losses,
            'val loss': v_losses
        },
        columns=['train loss', 'val loss']
    )
        #plot losses
        st.line_chart(df)

"""
build pipeline for spatial augmentations
"""
def build_transforms():
    trans = []
    if state.flip:
        x = monai.transforms.RandFlip(state.p_flip, state.spatial_axis)
        trans.append(x)
    if state.rotate:
        x = monai.transforms.RandRotate(range_x=state.range_x,range_y=state.range_y,prob=state.p_rotate )
        trans.append(x)
    if state.croping:        
        x = monai.transforms.RandSpatialCrop(state.roi_scale,max_roi_size=state.max_roi_scale)
        trans.append(x)
    if state.zoom:
        x = monai.transforms.RandZoom(prob=state.p_zoom,min_zoom=state.min_zoom, max_zoom=state.max_zoom)
        trans.append(x)
    trans = monai.transforms.Compose(trans)
    if len(trans)==0:
        return None
    return trans
"""
build pipeline for intensity augmentations
"""
def build_intensity_transform():
    trans = []  
    if state.one_off:
        x = monai.transforms.OneOf(
            [monai.transforms.RandGaussianNoise(prob=state.gaus_prop),
            monai.transforms.RandGibbsNoise(prob=state.p_gibs),

            ],(0.5,0.5)
        ) 
        trans.append(x)
    else:
        if state.gibbs_noise:
            x = monai.transforms.RandGibbsNoise(prob=state.p_gibs)
            trans.append(x)
        elif state.gauss_noise:
            x = monai.transforms.RandGaussianNoise(prob=state.gaus_prop)
            trans.append(x)
    if state.RandHistogramShift:
        x = monai.transforms.RandHistogramShift(prob=state.hist_prop, num_control_points=state.control_points)
        trans.append(x)
    if state.RandCoarseDropout:
        x = monai.transforms.RandCoarseDropout(prob=state.coarse_prop, holes=state.holes, spatial_size=state.hole_size)


    if len(trans)==0:
        return None




