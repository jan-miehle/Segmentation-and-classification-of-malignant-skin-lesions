
import unet

"""
loads an pretrained instance of unet
creates segmetation masks

network_file: pretrained model
save_path: dir to save sekmentation masks 
image_dir=None: If None, PH2 data set will be loaded from the expexted folder
size=512,  image site
channels=3: input channels have to be equal to pretrained model
layers=(64, 128, 256, 512, 1024): layers of model, have to be equal to pretrained model
strides=(2,2,2,2): stride size of model, have to be equal to pretrained model
"""
def segment(network_file, save_path, image_dir=None, size=512, channels=3, layers=( 64, 128, 256, 512, 1024), strides=(2,2,2,2)):
    network = unet.unet(img_dir=None,gt_dir=None,save_path=None,network_name=None, size=size, channels=channels, drop=0, layers=layers, strides=strides, jac=False)
    network.load_network(network_file)
    network.seg(image_dir,save_dir=save_path, channels=channels)
  
