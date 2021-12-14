import monai
import math
from monai.transforms import (
    Compose,
    AsChannelFirst,
    AsChannelLast,
    ToNumpy,
    FillHoles,
    LoadImage,
    AsDiscrete,
    Resize
    
)
#network architecture:
from monai.networks.nets import UNet as u_net
import monai.losses
import monai.optimizers
import matplotlib.pyplot as plt
from monai.transforms.intensity.array import NormalizeIntensity

import torch
import torch.utils
import torch.utils.data
import streamlit as st
import numpy
import os
import cv2
from dataset_unet_train import data_set_u_net as data_set_u_net
from data_set_unet_segmentation import data_set_segmentation as data_set_segmentation

"""
instance of unet network
performes train and test loop


img_dir=None : if none, ph2 dataset will be loaded
gt_dir=None : if none, ph2 dataset will be loaded
save_path ="saves",
network_name="network", 
size=128: image size
channels=3: input channels
drop=0.0 : dropout rate
layers=( 64, 128, 256, 512, 1024): network lyers
strides=(2,2,2,2): stride size
jac=False: use jaccard index? 
loss_function=None: If None, moai.losses.DiceCe is ued 
spatial_transform=None (monai.transforms): for web interface 
intensity_transform=None (monai.transforms): for web interface 
web=False: for web interface 

"""
class unet:
    def __init__(self, img_dir=None,gt_dir=None,save_path ="saves",network_name="network", size=128, channels=3, drop=0.0, layers=( 64, 128, 256, 512, 1024), strides=(2,2,2,2), jac=False, loss_function=None, spatial_transform=None, intensity_transform=None, web=False):
        self.web=web
        self.intensity_transform = intensity_transform
        self.spatial_transform = spatial_transform
        #streamlit output:
        if self.web:
            st.write('start of training..')
            st.write("network config: channels: ", channels, " layers:" , layers, " dropout:", drop, " jaccard =", jac)


        print(network_name)        
        print("network config: channels: ", channels, " layers:" , layers, " dropout:", drop, " jaccard =", jac)
   
      
        self.channels = channels
   

        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.size = size
        self.save_path = save_path  
        if self.save_path is not None:
            print("save to: ", self.save_path)
            os.makedirs(self.save_path, exist_ok=True)


        self.network_name = network_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.last_network= None

    	#instantiate u-net
        self.net =u_net(                      
            dimensions=2,
            in_channels=self.channels,
            out_channels=1,
                       channels=layers, 
            strides=strides, 
            num_res_units=2,
            dropout=drop,
            bias=True,
        ).to(self.device)
        
        # losses and optimizer:    
        
        if (loss_function == None):
            self.loss_fn= monai.losses.DiceCELoss(include_background=True, jaccard=jac,  squared_pred="True")
        else:
             self.loss_fn=loss_function
        print(self.loss_fn)
        if self.web:
            st.write("loss function:")
            st.write(self.loss_fn)
        self.optimizer = monai.optimizers.Novograd(self.net.parameters(), lr=0.001)



    def get_network_config(self):
        print(self.net)

    """
    shows example of image augmentation
    """
    def get_image_preview(self, img_save_path=None):
        preview_data =  data_set_u_net(self.img_dir, size=self.size, channels=self.channels, spatial_transform=self.spatial_transform, intensity_transform=self.intensity_transform)
        preview_loader = monai.data.DataLoader(preview_data, batch_size=1, num_workers=0, shuffle=False )
        for _, (X, y,  image_file, _, _) in enumerate(preview_loader):
            X = X.to("cpu")
            y = y.to("cpu")
            #img_trans =Compose([ ToNumpy(), AsChannelLast(), ])
            if (self.channels ==1):
                print("monochrome")
                trans = Compose([
                    LoadImage(image_only=True),  
                                    
                    ToNumpy(),
                ])   
            else:
                trans= Compose(
                    [
                    LoadImage(image_only=True), 
                    AsChannelFirst(),       
                    Resize([ self.size, self.size]), 
                    AsChannelLast(), 
                    ToNumpy()
                ]
                )           

            image = trans(image_file) 

            if self.channels==1:
                image = cv2.resize(image[0], (self.size, self.size))
            else:
                image = image[0]

            file_name = str(image_file).split('.')[0]
            file_name = file_name.split('/')
            if (len(file_name) > 1):
                file_name = file_name[len(file_name) -1]
            title = "" +str(file_name) 

            plt.clf()
            fig=plt.figure()
            fig.add_subplot(1,2,1)
            plt.axis('off')
            plt.imshow(image.astype(numpy.uint8))
            plt.title(title)

            #ergebnis
            fig.add_subplot(1,2,2)
            trans = Compose([AsChannelFirst(), ToNumpy()])
            reverse = Compose([NormalizeIntensity()])
            X = trans(X[0])
            X = reverse.inverse(X)
            p= trans(X)
            plt.axis('off')
            plt.imshow(p, cmap="binary")  
            
            if img_save_path is not None:     
                os.makedirs(img_save_path, exist_ok=True) 
                plt.savefig(img_save_path +"/" + title+".png", dpi=300, bbox_inches='tight',pad_inches = 0) 
            plt.show()
            plt.close() 
        



    """
    read images filesto data loader
    batch_size=10, 
    selection="all", if "hairs", selection of PH2 with hairs will be loaded, if "clean" folder of images without hairs will be loaded
    """
    def get_data(self, batch_size=10, selection="all"):
     

        self.all_data = data_set_u_net(self.img_dir, size=self.size, channels=self.channels,selection=selection, spatial_transform=self.spatial_transform, intensity_transform=self.intensity_transform)
        train_size = int(len(self.all_data)*0.6)
        val_size = int(len(self.all_data)*0.2)
        test_size = len(self.all_data) -train_size -val_size


        self.training_set, self.val_set, self.test_set = torch.utils.data.random_split(self.all_data,[train_size,val_size, test_size], generator=torch.Generator().manual_seed(98))   #seed paper: 42
     
        
        self.train_loader = monai.data.DataLoader(self.training_set, batch_size=batch_size, num_workers=0, shuffle=True )
        self.val_loader = monai.data.DataLoader(self.val_set, batch_size=batch_size, num_workers=0, shuffle=True)
        self.test_loader = monai.data.DataLoader(self.test_set, batch_size=1, num_workers=0)#,  collate_fn= monai.data.utils.pad_list_data_collate
        print("Traing set:", len(self.training_set), " elements")    
        print("Validation set:", len(self.val_set), " elements")    
        print("Test set:", len(self.test_set), " elements")    
    

    """
    start traing loop for self.net
    """
    def train(self, epochs=5):
        losses = []
        v_losses = []        
       
     
        min_loss = math.inf
        min_test_loss = math.inf
        for e in range(epochs):
            t_counter = 0          
            t_meanloss = 0.0
            #test-run:
            self.net.train()
            for _, (X, y,  _, _, orig_gt) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.net(X.float())


                loss =self.loss_fn(pred.float(),y.float() ) 

                # Backpropagation
                self.optimizer.zero_grad()            
                loss.backward()
                self.optimizer.step()


                t_meanloss = t_meanloss + loss.item()
                t_counter = t_counter +1                
            
               
            t_meanloss = t_meanloss/t_counter
            print("epoch ", (e+1),"off", epochs, f"mean loss: {t_meanloss:>7f}" )
            if self.web:
                st.write("epoch ", (e+1),"off", epochs, f"mean loss: {t_meanloss:>7f}" )

            losses.append(t_meanloss)

            #evaluation set:
            self.net.eval()            
            v_meanloss = 0.0
            counter = 0
            no_min = True
            for _, (_, _,  _, orig_image, orig_gt) in enumerate(self.val_loader):
                
                counter = counter + 1
                X = orig_image.to(self.device)  
                y = orig_gt.to(self.device)   
              
                pred = self.net(X.float())        

                loss =self.loss_fn(pred.float(),y.float())

                v_meanloss  = v_meanloss  + loss.item()
           
            v_meanloss = v_meanloss/counter
            #if new min val-loss isreached:
            if v_meanloss < min_loss:
                #test-run:
                test_loss = self.test(save=True)
                if test_loss < min_test_loss:
                    min_test_loss=test_loss
                    min_loss = v_meanloss
                    x = e+1
                    print("#################")
                    print("Minima at epoch", x, "loss:", v_meanloss )

                    #streamlit output:
                    if self.web:
                        st.write("Minimum validation loss at epoch", x, "loss:", v_meanloss )
                        st.write("testloss", test_loss)
                        no_min=False
                    
                    ## print loses
                    name = "epoch" + str(x) + "testloss" + str(test_loss) +"val_loss" + str(min_loss)
                    name = self.save(name)
                    #save actual state
                    if self.last_network is not None:
                        if os.path.exists(self.last_network ):
                            os.remove(self.last_network )
                    self.last_network = name
                    print("#################")


            v_losses.append(v_meanloss)
            print("epoch ", (e+1),  f"mean validation loss: {v_meanloss:>7f}") 

            #streamlit output:
            if self.web and no_min:
                st.write("epoch ", (e+1),  f"mean validation loss: {v_meanloss:>7f}") 



        ###### plot mean losses
        x = [i for i in range(epochs)]
        plt.clf()
        plt.plot(x, losses,  label = "train_loss")
        plt.plot(x, v_losses,  label = "val_loss")
        plt.xlabel("epoch")
        plt.ylabel("mean error")
        plt.yticks(numpy.arange(0, 1.1, 0.1))
        plt.legend()
        if self.save_path is not None:        
            plot_name= self.network_name + str(epochs) +".png"
            plt.savefig(plot_name)  
        plt.show()
        plt.close()         
        

        return x, losses, v_losses

    """
    test-loop
    save=False: If True, save image, prediction, thresholded prediction and gt
    """
    def test(self, save=False):       
        test_loss = 0
        test_post_loss = 0
        plotnumber = 0
        self.net.eval()
        test_counter =0
        for _, (_, _, image_file, orig_trans, orig_gt) in enumerate(self.test_loader):
                X = orig_trans.to(self.device)  
                y = orig_gt.to(self.device)    
                
                pred = self.net(X.float())        

                loss_vall = self.loss_fn(pred, y).item()
                #thresholding and cleaning:
                post_pred = self.postprocessing(pred)
              

                test_loss += loss_vall
                test_counter = test_counter + 1
                #save plot:
                if save:
                    self.test_set_plot(X ,y,pred, image_file, loss_vall,post_pred, plotnumber)
                plotnumber = plotnumber +1
                print(image_file," loss: ", loss_vall)

        test_loss /= test_counter
        test_post_loss /= test_counter
        print(f"Test loss: {test_loss:>7f}")

        return test_loss
   
    
    """
    creates segmentation mask in pretrained model
    image_dir: Folder, images to segment
    save_dir
    size: imagesize/ size to scale
    channels: channels of image: 1,3 or 4
    """
    def seg(self, image_dir, save_dir,size=512, channels=3):
        os.makedirs(save_dir, exist_ok=True)
        #load data:
        data = data_set_segmentation(image_dir, size=size, channels=channels,selection=None)
        loader = monai.data.DataLoader(data, batch_size=1, num_workers=0, shuffle=False)
        print(len(loader))
        self.net.eval()
        #segemt:
        for _, (X, image_file) in enumerate(loader):
            X = X.to(self.device) 
            X =self.net(X)
         
        
            trans = Compose([
                        ToNumpy(),
                    ])   
            #save:
            X =self.postprocessing(X)
            X = trans(X[0][0])
            file_name = image_file[0]
            file_name = file_name.split("/")[len(file_name.split("/"))- 1]
            file_name = file_name.split(".")[0]
            file_name = file_name + ".png"
            print("save:", file_name)
            b =cv2.imwrite(os.path.join(save_dir , file_name),  numpy.array(X).astype(int) *255,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(b)
       

            
    """
    safes network tp /netowrks/self.network.name or
    name (str)
    """
    def save(self, name=""):
        path = "networks/"+self.network_name
        os.makedirs(path, exist_ok=True)
        name = path +"/" +name+ "_network"
        torch.save(self.net.state_dict(), name)
        return name

    """
    postprocessing fpr prediction
    by Monai.transorms
    Thresholding
    FillHoles
    """
    def postprocessing(self, pred):
        trans = Compose([
            AsDiscrete(num_classes=2,threshold_values=True,logit_thresh=0.5),            
            FillHoles(),
        ])
        pred = pred.to(self.device)
        pred =trans(pred)
        return pred


    """
    plots image, prediction, thresholded prediction and gt for images in test-set
    saves plot to self.save_path
    """
    def test_set_plot(self, X, y, pred, image_file, loss, post_pred, plotnumber):        
            X = X.to("cpu")
            y = y.to("cpu")
            pred= pred.to("cpu")
            if (self.channels ==1):
                trans = Compose([
                    LoadImage(image_only=True),                    
                    ToNumpy(),
                ])   
            else:
                trans= Compose(
                    [
                    LoadImage(image_only=True), 
                    AsChannelFirst(),       
                    Resize([ self.size, self.size]), 
                    AsChannelLast(), 
                    ToNumpy()
                ]
                )           

            mask_trans = Compose([ToNumpy()])
            image = trans(image_file) 

            if self.channels==1:
                print(image)
                image = cv2.resize(image[0], (self.size, self.size))
            else:
                image = image[0]

            file_name = str(image_file).split('.')[0]
            file_name = file_name.split('/')
            if (len(file_name) > 1):
                file_name = file_name[len(file_name) -1]
            l = round(loss, 3)
            title = "" +str(file_name) + ", segmentation-loss= " + str(l) + "."
            plt.clf()
            fig=plt.figure()
            fig.add_subplot(1,4,1)
       
            plt.axis('off')
            plt.imshow(image.astype(numpy.uint8))
            plt.title(title)

            #ergebnis
            fig.add_subplot(1,4,2)
            p= mask_trans(pred[0][0])
            plt.axis('off')
            plt.imshow(p, cmap="binary")  
            #ergebnis > 0.5
            fig.add_subplot(1,4,3)
            p = mask_trans(post_pred[0][0])
            plt.axis('off')
            plt.imshow(p, cmap="binary")       
            #gt:
            fig.add_subplot(1,4,4)
            y = mask_trans(y[0][0])
            plt.axis('off')
            plt.imshow(y, cmap="binary")
    
            if self.save_path is not None:      
                plt.savefig(self.save_path +"/" + str(plotnumber)  +"_plot.png", dpi=300, bbox_inches='tight',pad_inches = 0) 
            plt.show()
            plt.close() 

    """
    load pretrained model from network_file
    """
    def load_network(self, network_file):
        self.net.load_state_dict(torch.load(network_file, map_location=torch.device(self.device)))
      
        
    
