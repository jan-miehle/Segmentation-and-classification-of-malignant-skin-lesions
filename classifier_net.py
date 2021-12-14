import monai
import math
from monai.transforms import ( Compose,
    AsDiscrete,
    Compose,
    LoadImage,
    Resize,
    AsChannelFirst,
    ToNumpy,
    AsChannelLast, 
)
import monai.metrics 
import monai.data
import monai.networks.nets  
import monai.optimizers
import matplotlib.pyplot as plt

import classifier_train_data, classifier_test_data
import torch
import os
import numpy
from pathlib import Path

"""
Class for creating and traing the classifier network
image_files :list of image files for training set
test_files : list of image files for test set
mask_files : list of mask files for training set
test_mask_files  : list of mask files for test set
label_csv : label file for training
test_label_csv  : label file for testset
size=512 : image size for rescaling
network_name="classifier"
network=None : Network class, if None monai.networks.nets.Classifier will be used
use_noise=True : use intensity augmentations or not
drop=0.0 dropout rate
weight=1 factor to multiply with pos_weight 
threshold=0.5: classification thresholf
"""
class classifier_network:
    def __init__(self, image_files,test_files, mask_files, test_mask_files, label_csv,test_label_csv, size=512, network_name="classifier", network=None, use_noise=True, drop=0.0, weight=1, threshold=0.5): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.network_name = network_name
        self.image_files = image_files
        self.test_files = test_files
        self.mask_files = mask_files
        self.test_mask_files = test_mask_files
        self.label_csv = label_csv
        self.test_label_csv = test_label_csv
        self.size = size
        self.use_noise = use_noise
        self.weight = 4.20231213872 * weight
        if self.weight == 0:
            self.weight=1
        self.threshold =threshold
        print("pos weight:", self.weight)
        ## for testing different thresholds ineach epoch:
        self.threshold_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print(self.mask_files)
        print(self.test_mask_files)

        
        #https://docs.monai.io/en/latest/networks.html#classifier       
        if network == None:            
            self.net = monai.networks.nets.Classifier(            
            in_shape=(3, size,size), 
            classes=1,
            channels=(64, 128, 256, 512),
            strides = (2,2, 2),
            dropout = 0.2,
            act='PRELU',   
            kernel_size=3, 
            num_res_units=2,
            ).to(self.device)
        else:
            self.net=network.to(self.device)        
        
        #according to https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb : 
        self.loss_fn= torch.nn.BCELoss(reduction='none') 
        #different loss-fkt for validation/test because there is no weighting of loss values   
        self.val_loss= torch.nn.BCELoss()
        self.optimizer =torch.optim.Adam(self.net.parameters(), lr=0.001)  


    """
    prints network config
    """
    def get_network_config(self):
        print(self.net)
  



    """
    loads the data-sets. 
    must be called before training
    """
    def get_data(self, batch_size=4):
        self.batch_size = batch_size
 
        self.training_set  = classifier_train_data.train_data(img_files = self.image_files,labels= self.label_csv, mask_files= self.mask_files, size=self.size, use_noise=self.use_noise)
        self.val_test_set =classifier_test_data.test_data(img_files=self.test_files, labels=self.test_label_csv, mask_files=self.test_mask_files, size=self.size)

        val_size = int(len(self.val_test_set) * 0.3)
        test_size = len(self.val_test_set) - val_size
        print(val_size)
        print(test_size)

        #split testset in train and validation set
        self.validation_set, self.test_set = torch.utils.data.random_split(self.val_test_set,[val_size,test_size], generator=torch.Generator().manual_seed(98))   

        self.train_loader = monai.data.DataLoader(self.training_set, batch_size=batch_size, num_workers=0, shuffle=True )
        self.val_loader = monai.data.DataLoader(self.validation_set, batch_size=batch_size, num_workers=0, shuffle=False)
        self.test_loader = monai.data.DataLoader(self.test_set, batch_size=batch_size ,num_workers=0, shuffle=False)
        

    """
    start training
    epochs: number of training epochs
    search_treshhold=True : if true, a list of possible thresholds will be tested at each minimum validation loss
    
    """
    def train(self, epochs=5, search_treshhold=True):
        ###saves scores from each epoch:
        losses = []
        v_losses = []
        
        prec_score = []
        acc_score =[]
        rec_score =[]
        fnr_score=[]
        fpr_score = []
        tpr_score = []
        tnr_score = []
        f1_score = []
        #for storing best f-scores
        f1_max = -math.inf
        f1_max_epoch = 0
        self.best_f_score_treshhod= (0.0,0.0, math.inf)
        for e in range(epochs):
            print("epoch", e+1, " off ", epochs)
            t_meanloss = 0.0
            counter = 0
            
            self.net.train()
            #training step:
            for batch_numer, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)       



                pred = self.net(X.float()) 
                s = torch.sigmoid(pred)                
                loss =self.loss_fn(s,y.float())

                #weight loss:
                a = torch.ones((len(s),1)).double().to(self.device)      
                b = (y * self.weight).to(self.device)               
                w= torch.where(b > 0, b, a).double() # w = self.weight where y=1 else 1
                loss = loss*w
                loss = torch.mean(loss)
       
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #count losses
                t_meanloss = t_meanloss + loss.item()
                counter = counter + 1                

                ##print feedback
                if batch_numer % 90 == 0:
                    step = batch_numer * self.batch_size
                    print(f"mean loss: {loss.item():>4f} in step {step:>5d} of {len(self.train_loader.dataset):>5d}")
            t_meanloss = t_meanloss / counter
            losses.append(t_meanloss)
            print("epoch ", (e+1),"mean loss: ", t_meanloss )

            
            
            #eval:
            self.net.eval()
            v_meanloss = 0.0
            v_counter = 0
            matrix = None
            # create empty confusion matrix:
            self.matrix_list= self.init_treshholeded_values()  #für opt. f-score
            self.first_run = True
            # count positiv examples in validation set:
            positives_val = 0
            for batch_numer, (X, y, img_file) in enumerate(self.val_loader):
                X = X.to(self.device) 
                y = y.to(self.device)
                pred = self.net(X.float()) 
                pred = torch.sigmoid(pred)
              

                loss =self.val_loss(pred,y.float()) 
                # count poitiv examples in validation set:
                positives_val = positives_val + torch.sum(y)


                v_meanloss = v_meanloss + loss.item()
                v_counter = v_counter + 1
                #transform for thresholding :
          
                post = Compose([AsDiscrete(num_classes=1, threshold_values=True, logit_thresh=self.threshold )])  #, logit_thresh=0.3    # DenseNet hat negative loss wertr
                #test different thresholds
                if search_treshhold:
                        self.calc_treshholded_rates(pred, y)
                #thresholding_
                pred_discrete = post(pred)
               

                #create one confusion matrix for each element:
                m = monai.metrics.get_confusion_matrix(pred_discrete, y)  
                n = m[0]  
                for i in range(1, len(X)):
                    n = n + m[i]           
                
                if matrix== None:
                    matrix = n
                else:
                    matrix = matrix + n
            print("positive examples vaildation set:", positives_val)

              

            #collect metrics:
            v_meanloss = v_meanloss / v_counter
            v_losses.append(v_meanloss)
            
            print(matrix)
            fnr = monai.metrics.compute_confusion_matrix_metric("fnr", matrix)
            fpr =  monai.metrics.compute_confusion_matrix_metric("fpr", matrix)
            tpr =  monai.metrics.compute_confusion_matrix_metric("sensitivity", matrix)
            tnr = monai.metrics.compute_confusion_matrix_metric("specificity", matrix)
            prec =  monai.metrics.compute_confusion_matrix_metric("precision", matrix)
            acc = monai.metrics.compute_confusion_matrix_metric("acc", matrix)
            rec = monai.metrics.compute_confusion_matrix_metric("recall", matrix)
            f1  = monai.metrics.compute_confusion_matrix_metric("f1 score", matrix)
            print("epoch ", (e+1),"mean validation loss: ", v_meanloss )
            
            print("sensitivity (tpr): ",tpr)            
            print("specificity (tnr): ", tnr)
            print("miss rate(fnr): ", fnr)
            print("fall out (fpr) : ",fpr)            
            print("precision: ",prec)
            print("recall: ", rec)
            print("accuracy: ", acc)
            print("f1", f1) 
            fnr_score.append(fnr)
            fpr_score.append(fpr)
            tpr_score.append(tpr)
            tnr_score.append(tnr)

            prec_score.append(prec)
            acc_score.append(acc)
            rec_score.append(rec)
            f1_score.append(f1)

            ###eval f-score
            if f1 > f1_max:
                f1_max = f1
                f1_max_epoch =e+1
                print("####################")
                print("new best f1 score=", f1_max, " in epoch ", f1_max_epoch, " for validation set")
                print("####################")
                test_score = self.test()
                name = self.network_name + str(e+1)
                self.save(name)

            if search_treshhold:
                self.show_treshholded_metrics()
            else:
                if f1 > self.best_f_score_treshhod[0]:
                     self.best_f_score_treshhod= (f1, rec, prec, self.threshold)

            print("best fscore, recall, treshhold:", self.best_f_score_treshhod)

               
                

        if search_treshhold:
            self.show_treshholded_metrics()
        print("best f1 score=", f1_max, " in epoch ", f1_max_epoch)
        print("with other treshhold", self.best_f_score_treshhod[0],"recall: ", self.best_f_score_treshhod[1], "precision ",self.best_f_score_treshhod[2], " treshold: ", self.best_f_score_treshhod[3])
        self.test()





            
            
    ###### plot mean losses
        x = [i for i in range(epochs)]
        plt.clf()
        plt.plot(x, losses,  label = "train_loss")
        plt.plot(x, v_losses,  label = "val_loss")
        plt.xlabel("epoch")
        plt.ylabel("mean error")
        plt.legend()        
        plot_name= self.network_name  +"_losses.png"
        plt.savefig(plot_name)  
        plt.show()
        plt.close() 

        #plot tpr etc
        x = [i for i in range(epochs)]
        plt.plot(x, tpr_score, label="tpr")
        plt.plot(x, tnr_score, label="tnr")
        plt.yticks(numpy.arange(0, 1.1, 0.1))
        plt.xlabel("epoch")
        plt.ylabel("rate")
        plt.legend()
        plot_name= self.network_name  +"_metrics.png"
        plt.savefig(plot_name)  
        plt.show()
        plt.close() 

        #plot accurray f-score
        x = [i for i in range(epochs)]
        plt.plot(x, acc_score, label="accuracy")
        plt.plot(x, f1_score, label="F-score")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.yticks(numpy.arange(0, 1.1, 0.1))
        plt.legend()
        plot_name= self.network_name  +"_acc.png"
        plt.savefig(plot_name)  
        plt.show()
        plt.close() 




        #plot recall precision
        x = [i for i in range(epochs)]
        plt.plot(x, prec_score, label="precision")
        plt.plot(x, rec_score, label="recall")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.yticks(numpy.arange(0, 1.1, 0.1))
        plt.legend()
        plot_name= self.network_name  +"_recall.png"
        plt.savefig(plot_name)  
        plt.show()
        plt.close() 


    """
    returns empty list for confusion matrices
    """
    def init_treshholeded_values(self):
        matrix_list = []
        return matrix_list

    """
    Calc confusion matrices for different threshholds
    """
    def calc_treshholded_rates(self,pred, y):
        i = 0
        for treshhold in self.threshold_values :
            post = Compose([AsDiscrete(num_classes=1, threshold_values=True, logit_thresh=treshhold)])
            x =  post(pred)
            

            matrix_a =  monai.metrics.get_confusion_matrix(x,y)

            n = matrix_a[0]
            for j in range(1, len(matrix_a)):
                n = n +  matrix_a[j]                     

            if self.first_run:
                self.matrix_list.append(n)
            else:           
                self.matrix_list[i] = self.matrix_list[i] + n

            i = i +1
        self.first_run = False
    """
    shows confusion matrix metrics for different thresholds
    
    """
    def show_treshholded_metrics(self):
        i = 0
        print(self.matrix_list)
        for treshhold in  self.threshold_values:
            matrix = self.matrix_list[i]
            print("optimiert mit trehhold ", treshhold)
            f1_f =  monai.metrics.compute_confusion_matrix_metric("f1 score", matrix)
            prec_f =  monai.metrics.compute_confusion_matrix_metric("precision", matrix)
            rec_f =  monai.metrics.compute_confusion_matrix_metric("recall", matrix)
            print("f1 score: ", f1_f)
            print("precission ", prec_f, " recall: ", rec_f)
            print("################################################")
            if f1_f > self.best_f_score_treshhod[0]:
                self.best_f_score_treshhod = (f1_f,rec_f, prec_f, treshhold)
            i = i+1

    """
    run test set
    """
    def test(self):
            #eval:
            self.net.eval()
            test_meanloss = 0.0
            test_counter = 0
            matrix = None
            
            self.first_run = True
            positives_val = 0
            for _, (X, y, _) in enumerate(self.test_loader):
                X = X.to(self.device) #!!!!!!!
                y = y.to(self.device)
                pred = self.net(X.float()) 
                pred = torch.sigmoid(pred)              

                loss =self.val_loss(pred,y.float()) 

                positives_val = positives_val + torch.sum(y)

                test_meanloss = test_meanloss + loss.item()
                test_counter = test_counter + 1
                """matrix:"""          
                post = Compose([AsDiscrete(num_classes=1, threshold_values=True, logit_thresh=self.threshold )])  #, logit_thresh=0.3    # DenseNet hat negative loss wertr

                pred_discrete = post(pred)


                m = monai.metrics.get_confusion_matrix(pred_discrete, y)  
                n = m[0]  
                for i in range(1, len(X)):
                    n = n + m[i]           
                
                if matrix== None:
                    matrix = n
                else:
                    matrix = matrix + n
            f1_test =  monai.metrics.compute_confusion_matrix_metric("f1 score", matrix)
            prec_test =  monai.metrics.compute_confusion_matrix_metric("precision", matrix)
            rec_test =  monai.metrics.compute_confusion_matrix_metric("recall", matrix)
            print("--------------------------------------------***---------------------------------------------")
            print("testset f_score: ", f1_test, " precision: ", prec_test, " recall ", rec_test)
            return f1_test, prec_test, rec_test


    """
    saves network to networks/classifier/
    name: filename
    returns save_path
    
    """
    def save(self, name):
        path = "networks/classifier/"+self.network_name
        os.makedirs(path, exist_ok=True)
        name = path +"/" +name+ "_network"
        torch.save(self.net.state_dict(), name)
        return name

    """
    plots example images for image augmentation
    img_save_path=None : If not none, plots will be saved to this path
    
    """
    def get_image_preview(self, img_save_path=None):
        preview_data =  classifier_test_data.test_data(img_files = self.image_files,labels= self.label_csv, mask_files= self.mask_files, size=self.size)
        preview_loader = monai.data.DataLoader(preview_data, batch_size=1, num_workers=0, shuffle=False )
        for _, (X, _, img_file) in enumerate(preview_loader):
            X = X.to("cpu")[0]
            trans= Compose(
                    [
                    LoadImage(image_only=True), 
                    AsChannelFirst(),       
                    Resize([ self.size, self.size]), 
                    AsChannelLast(), 
                    ToNumpy()                               
                ]
                )   

            file_name = str(img_file).split('.')[0]
            file_name = file_name.split('/')
            if (len(file_name) > 1):
                file_name = file_name[len(file_name) -1]
            title = "" +str(file_name) 
            plt.clf()
            fig=plt.figure()
            fig.add_subplot(1,2,1)

            image =trans(img_file)[0]
            print(image.shape)
            plt.axis('off')
            plt.imshow(image.astype(numpy.uint8))
            plt.title(title)
            #ergebnis
            fig.add_subplot(1,2,2)
            trans = Compose([AsChannelLast(), ToNumpy()])
            p= trans(X)
            plt.axis('off')
            plt.imshow(p)              
            if img_save_path is not None:     
                os.makedirs(img_save_path, exist_ok=True) 
                plt.savefig(img_save_path +"/" + title+".png", dpi=300, bbox_inches='tight',pad_inches = 0) 
            plt.show()
            plt.close() 
########################################################################################################################



"""
Instantiates classifier_network
reads image list from isic folders before

size=512 : image size for rescaling
name : network name/ file name
use_gt (bool): if False, images will not be masked
network=None : Network class, if None monai.networks.nets.Classifier will be used
weight=1 factor to multiply with pos_weight 
use_noise=True : use intensity augmentations or not
threshold=0.5: classification threshold
gt_dir="ISIC/train_gt" : standart folder for train gt
test_gt_dir= 'ISIC/gt_test' : standart folder for test gt
drop=0.0 dropout rate

"""
def create_network_isic(size, name ,use_gt, use_noise=True, drop=0.0, network=None, weight=1, threshold=0.5, gt_dir="ISIC/train_gt", test_gt_dir= 'ISIC/gt_test'):
    image_dir = 'ISIC/ISBI2016_ISIC_Part3B_Training_Data'
    test_dir = "ISIC/ISBI2016_ISIC_Part3B_Test_Data"
    image_files = []
    gt_files= []
    for child in Path(image_dir).iterdir():
        img = image_dir +"/" + child.name
        image_files.append(img)
    for child in Path(gt_dir).iterdir():
        gt = gt_dir+"/" + child.name
        gt_files.append(gt)
    image_files.sort()
    gt_files.sort()

    test_image_files = []
    test_gt_files = []
    for child in Path(test_dir).iterdir():
        img = test_dir +"/" + child.name
        test_image_files.append(img)
    for child in Path(test_gt_dir).iterdir():
        gt = test_gt_dir+"/" + child.name
        test_gt_files.append(gt)
    test_image_files.sort()
    test_gt_files.sort()




    if use_gt == False:
        gt_files = None
    label_file = "ISIC/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
    test_label_file = str("ISIC/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv")
    net = classifier_network(image_files,test_image_files, gt_files, test_gt_files, label_file,test_label_file, size, network_name=name, use_noise=use_noise, drop=drop, network=network, weight=weight, threshold=threshold)

    return net




