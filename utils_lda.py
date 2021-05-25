# -*- coding: utf-8 -*-
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import random
from torchvisionnewtransforms import RandomPerspective,ColorJitter,RandomAffine,RandomRotation,RandomHorizontalFlip,RandomVerticalFlip
from scipy.optimize import linear_sum_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle

#######################################################
# Evaluate Critiron
#######################################################
def augment(x_tr):

        
        x_tr = transforms.ToPILImage()(x_tr)
        x_tr = RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)(x_tr)
        x_tr = RandomHorizontalFlip(p=0.5)(x_tr)
        x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        #x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)(x_tr)
        #x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        x_tr = RandomRotation(30, resample=False, expand=False, center=None)(x_tr)
        x_tr = transforms.ToTensor()(x_tr)
        

        return x_tr

def hard_mine_classes(out_score, prob_threshold):
    """
    Find a ratio k of samples that produce the highest propability.    
    These samples will be used to finetune the network.
    Return the indexes and predicted labels of corresponding images.
    
    """
    y_score = torch.from_numpy(out_score)
    softmax_convert = torch.nn.Softmax(dim =1)
    y_prob = softmax_convert(y_score)    
    _, y_prob_label = y_prob.max(1)
    n_classes = y_prob.size(1)
    sample_idx_each_class = []
    k_labels = []
    k_indexes = []
    c_k_indexes = []
    for i in range(n_classes):
        #print('i = ' ,i)
        t = (y_prob_label==i).nonzero()
        sample_idx_each_class.append(t)
        sample_idx_each_class[i] = sample_idx_each_class[i][:,0] #To convert tensor of multiple tensor to numpy array
   
    #print('sample_idx_each_class[i]= ', sample_idx_each_class)      
    #print('sample_c1_idx = ', sample_c1_idx)      
    #print(' y_prob = ' , y_prob)
    #print('y_prob_label = ' , y_prob_label)
    #print('ypred[sample_c0_idx,0] = ', y_prob[sample_c0_idx,0])
    class_sizes = []
    for i in range(n_classes):
        class_sizes.append(sample_idx_each_class[i].size(0))     
    #print('class_sizes =', class_sizes)  

   
    #print('k_value = ' ,k_value)
    
    for i in range(n_classes):
        samples_class_i = sample_idx_each_class[i]
        k_indexes_i = (y_prob[samples_class_i,i] > prob_threshold).nonzero()[:,0]
        k_indexes.append(k_indexes_i)
        #print('k0_indexes =', k0_indexes)    
        c_k_indexes_i = samples_class_i[k_indexes_i]
        c_k_indexes.append(c_k_indexes_i )
        #print('c0_k0_indexes =', c0_k0_indexes)
        #k_labels_i = y_prob_label[c_k_indexes_i]
        #k_labels.append(k_labels_i)
        #print('ypred[sample_c1_idx,1] = ', y_prob[sample_c1_idx,1])
    #print('c_k_indexes = ', c_k_indexes)   
    k_indexes = torch.cat(c_k_indexes)
    #print('k_indexes = ', k_indexes)
    #k_labels = y_prob_label[k_indexes]
    #print('k_labels = ', k_labels)      
    r_r = torch.randperm(k_indexes.size(0))
    
    hm_num = int(r_r.size(0)/1)
    r = r_r[:hm_num]
    #r = r_r
    #print('r = ',r)
    k_indexes = k_indexes[r]
    #print('k_indexes = ', k_indexes)
    k_labels = y_prob_label[k_indexes]
    k_labels = np.array(k_labels)
    k_labels = k_labels.astype(np.int64)
    
    return k_indexes.tolist(), k_labels


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    #print('y_true sizes ',y_true.shape)
    #print('y_pred sizes ',y_pred.shape)
    print('y_true sizes ',y_true.size)
    print('y_pred sizes ',y_pred.size)    
    #y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    rox_ind,col_ind = linear_sum_assignment(w.max() - w)  #Hungarian algorithm#############
    ind = np.stack((rox_ind,col_ind), axis = 1)
    #print('w matrix =',w)

    #print('index assignment: ',ind)
    assign_dict = {}
    for pair in ind:
        #print('pair =', pair)
        
        try:
            assign_dict[pair[0]].append(pair[1])
        except KeyError:
            assign_dict[pair[0]] = [pair[1]]
        #print('assign_dict =', assign_dict)    
    # Confusion matrix
    #adjusted_pred = y_pred
    adjusted_pred = [assign_dict.get(n) for n in y_pred]
    #print('adjusted_pred =', adjusted_pred)
    
    conf_mat=confusion_matrix(y_true, adjusted_pred)
    print('confusion matrix = ')
    print(conf_mat)

    perclass_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('perclass_accuracy =')
    print(perclass_accuracy)     
    
    if D ==2:
        target_names = ['Abnormal', 'Normal']
    elif D ==3:
        target_names = ['CLL', 'FL', 'MCL']
    elif D==7:
        target_names = ['Carcinoma', 'L_Dysplastic', 'M_Dysplastic', 'NColumnar', 'NIntermediate', 'NSuperficiel', 'S_Dysplastic']
    elif D==10:
        target_names =  ['Actin', 'DNA', 'Endosome', 'ER', 'Golgia', 'Golgpp', 'Lysosome','Microtubules','Mitochondria', 'Nucleolus']
    print(classification_report(y_true, adjusted_pred, target_names=target_names))
    
    #For multi-label classification problem: ROC curve should be adjusted
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def roc_curve_plot(y_true_1,y_score,n_classes,split_k):
    print('n_classes =',n_classes)
    #While you want to compute recall-precision for sklearn classifiers 
    #you cannot use general pipeline containing preprocessing.label_binarize
    #preprocessing.label_binarize for 2 classes gives single vector

    if n_classes== 2:
        classes_idx= np.arange(n_classes+1)
        # Binarize the output
        y_true = label_binarize(y_true_1, classes=classes_idx)[:,:-1] # classes='Abnormal','Normal' ONLY FOR BINARIZATION
        classes_names = ['Abnormal','Normal']
    elif n_classes== 3:
        classes_idx= np.arange(n_classes)
        y_true = label_binarize(y_true_1, classes=classes_idx)
        classes_names = ['CLL', 'FL', 'MCL'] 
    elif n_classes== 5:
        classes_idx= np.arange(n_classes)
        y_true = label_binarize(y_true_1, classes=classes_idx)
        classes_names = ['CLL', 'FL', 'MCL', 'Abnormal','Normal']        
    elif n_classes== 7:
        classes_idx= np.arange(n_classes)
        y_true = label_binarize(y_true_1, classes=classes_idx)
        classes_names = ['Carcinoma', 'L_Dysplastic', 'M_Dysplastic', 'NColumnar', 'NIntermediate', 'NSuperficiel', 'S_Dysplastic']
    elif n_classes== 10:
        classes_idx= np.arange(n_classes)
        y_true = label_binarize(y_true_1, classes=classes_idx)
        classes_names = ['Actin', 'DNA', 'Endosome', 'ER', 'Golgia', 'Golgpp', 'Lysosome','Microtubules','Mitochondria', 'Nucleolus']
        
    elif n_classes== 15:
        classes_idx= np.arange(n_classes)
        y_true = label_binarize(y_true_1, classes=classes_idx)
        classes_names = ['he_Actin', 'he_DNA', 'he_Endosome', 'he_ER', 'he_Golgia', 'he_Golgpp',  'he_Lysosome','he_Microtubules','he_Mitochondria', 'he_Nucleolus', 'lym_CLL', 'lym_FL', 'lym_MCL', 'pap_Abnormal','pap_Normal']
        
    #print('classes_idx =',classes_idx)
    #print('y_true : ', y_true)
    print('y_true shape = ', y_true.shape)
    print('y_score shape: ', y_score.shape)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print("i = ",i)
        print("fpr[i] = ", fpr[i])
        print("tpr[i] = ", tpr[i])
        print("roc_auc[i] = ", roc_auc[i])
        print('thresholds = ',thresholds[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    f = plt.figure(figsize=(8, 8))
    lw = 2
    '''
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    '''
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.3f})'
                 ''.format(classes_names[i], roc_auc[i]))
                 
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (AUC = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (AUC = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curves')
    plt.legend(loc='lower right',fancybox=True, shadow=True,fontsize='small')
  
   
    plt.savefig('./results/ROC_AUC___split_k'+ str(split_k)+'___.pdf')
    plt.close()
    
    
    y_score_as_torch = torch.from_numpy(y_score)
    softmax_convert = torch.nn.Softmax(dim =1)
    y_probs = softmax_convert(y_score_as_torch) 
    print(classification_report( np.argmax(y_true,axis = 1), np.argmax(y_probs,axis = 1), target_names=classes_names))
    return f, 



def load_papsmear(data_path='./dataset_papsmear'):

    #normalize = transforms.Normalize(mean=[0.5],std=[0.5])     
    
    train_dataset = datasets.ImageFolder(data_path, transforms.Compose([ #Transform only when get_item())
            #transforms.Grayscale(num_output_channels=1), #When using ImageFolder class and with no custom loader, pytorch uses PIL to load image and converts it to RGB
            transforms.Resize(224),
            #transforms.RandomSizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ])) 
    
    
    imgs_tuples = train_dataset.imgs
    
    data_tensor = None
    target_tensor = []
    for i in range  (917) : #(220) (86): # (400,491): (917): # (len(imgs_tuples)):
        #index = random.randint(1,860)
        index = i
        path,target = imgs_tuples[index]
        image = train_dataset.loader(path)
        #print('image shape tuple length',len(image))
        #print('image shape pre',image.size)
        #image = transforms.Grayscale(num_output_channels=1)(image)
        image = transforms.Resize((224,224))(image)
        #print('image shape post',image.size)
        #image = image.convert('RGB') 
        #image = transforms.functional.adjust_contrast(image,2)

        #image = transforms.fuctional.adjust_gamma(image,2)
        image = transforms.ToTensor()(image)
        #image = image+0.
        #image = transforms.Normalize(mean=[0.5],std=[0.5])(image)  
        #print('image shape',image.shape)
        if data_tensor is None:
            img = torch.unsqueeze(image, 0)   #For color image of N channels
            data_tensor = img
            target_tensor.append(target)
        else:
            img = torch.unsqueeze(image, 0)      #For color image of N channels      
            data_tensor = torch.cat((data_tensor, img),0)
            target_tensor.append(target)
        #print("image = {} target= {}".format(i,target))
        
    #data_tensor = data_tensor >0.2
    #data_tensor = data_tensor.float()
    #data_tensor = torch.squeeze(data_tensor,1)
    data_train = np.array(data_tensor)
    labels_train =   np.array(target_tensor)
    #print('labels_train = ',labels_train)
    data_train = data_train.astype(np.float32)
    #data_train = np.expand_dims(data_train,axis = 1) #to be processed by convolution layer
    #data_train = np.reshape(data_train,[-1,1,16,16])
    labels_train = labels_train.astype(np.int64)
    

    x = data_train
    #x = np.divide(x, 255.)
    y = labels_train
    print( 'papsmear samples', x.shape)
    #print('papsmear true labels',train_dataset.class_to_idx)
    return x, y

class papsmearDataset(Dataset):

    def __init__(self,x_train =None, y_train =None):
        
        if x_train is None and y_train is None:
            x_tr, y_tr= load_papsmear()
            x_tr = torch.from_numpy(x_tr)
        else:
            x_tr= x_train
            y_tr = y_train
            
        self.x_tr = x_tr
        #self.x_tr = resize_tensor(x_tr,28,28)
        #x_tr = torch.nn.functional.interpolate(x_tr,[28,28],mode='bilinear')
        #self.x_tr = np.divide(x_tr, 255.)

        #print('x_tr shape',self.x_tr.shape)
        #print('x 0 values',x_tr[0,0,5:15,5:15])
        print('x min values',torch.min(self.x_tr))
        print('x max values',torch.max(self.x_tr))
        
        self.y_tr = y_tr
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
        #x_tr = torch.from_numpy(np.array(self.x_tr[idx]))
        #x_tr = resize_tensor(x_tr,28,28)
        x_tr = self.x_tr[idx]
        
        x_tr = transforms.ToPILImage()(x_tr)
        x_tr = RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)(x_tr)
        x_tr = RandomHorizontalFlip(p=0.5)(x_tr)
        x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        #x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)(x_tr)
        #x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        x_tr = RandomRotation(30, resample=False, expand=False, center=None)(x_tr)
        x_tr = transforms.ToTensor()(x_tr)
        
        y_tr = torch.from_numpy(np.array(self.y_tr[idx]))
        idx = torch.from_numpy(np.array(idx))
        return x_tr,y_tr,idx
    
def load_hela(data_path='./Hela_jpg'):

    #normalize = transforms.Normalize(mean=[0.5],std=[0.5])     
    
    train_dataset = datasets.ImageFolder(data_path, transforms.Compose([ #Transform only when get_item())
            transforms.Grayscale(num_output_channels=1), #When using ImageFolder class and with no custom loader, pytorch uses PIL to load image and converts it to RGB
            transforms.Resize(224),
            #transforms.RandomSizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ])) 
    
    
    imgs_tuples = train_dataset.imgs
    
    data_tensor = None
    target_tensor = []
    for i in range  (862): # (862): # (len(imgs_tuples)):
        #index = random.randint(1,860)
        index = i
        path,target = imgs_tuples[index]
        image = train_dataset.loader(path)
        image = transforms.Grayscale(num_output_channels=1)(image)
        image = transforms.Resize((224,224))(image)
        image = image.convert('RGB') 
        #image = transforms.functional.adjust_contrast(image,2)

        #image = transforms.fuctional.adjust_gamma(image,2)
        image = transforms.ToTensor()(image)
        #image = image+0.
        #image = transforms.Normalize(mean=[0.5],std=[0.5])(image)  
        #print('image shape',image.shape)
        if data_tensor is None:
            img = torch.unsqueeze(image, 0)   #For color image of N channels
            data_tensor = img
            target_tensor.append(target)
        else:
            img = torch.unsqueeze(image, 0)      #For color image of N channels      
            data_tensor = torch.cat((data_tensor, img),0)
            target_tensor.append(target)
        #print("image = {} target= {}".format(i,target))
        
    #data_tensor = data_tensor >0.2
    #data_tensor = data_tensor.float()
    #data_tensor = torch.squeeze(data_tensor,1)
    data_train = np.array(data_tensor)
    labels_train =   np.array(target_tensor)
    #print('labels_train = ',labels_train)
    data_train = data_train.astype(np.float32)
    #data_train = np.expand_dims(data_train,axis = 1) #to be processed by convolution layer
    #data_train = np.reshape(data_train,[-1,1,16,16])
    labels_train = labels_train.astype(np.int64)
    

    x = data_train
    #x = np.divide(x, 255.)
    y = labels_train
    print( 'Hela samples', x.shape)
    #print('Hela true labels',train_dataset.class_to_idx)
    return x, y

class HelaDataset(Dataset):

    def __init__(self,x_train =None, y_train =None):
        
        if x_train is None and y_train is None:
            x_tr, y_tr= load_hela()
            x_tr = torch.from_numpy(x_tr)
        else:
            x_tr= x_train
            y_tr = y_train
        self.x_tr = x_tr
        #self.x_tr = resize_tensor(x_tr,28,28)
        #x_tr = torch.nn.functional.interpolate(x_tr,[28,28],mode='bilinear')
        #self.x_tr = np.divide(x_tr, 255.)

        #print('x_tr shape',self.x_tr.shape)
        #print('x 0 values',x_tr[0,0,5:15,5:15])
        print('x min values',torch.min(self.x_tr))
        print('x max values',torch.max(self.x_tr))
        
        self.y_tr = y_tr
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
        #x_tr = torch.from_numpy(np.array(self.x_tr[idx]))
        #x_tr = resize_tensor(x_tr,28,28)
        x_tr = self.x_tr[idx]
        
        x_tr = transforms.ToPILImage()(x_tr)
        #x_tr = RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)(x_tr)
        #x_tr = RandomHorizontalFlip()(x_tr)
        #x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        #x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(x_tr)
        #x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        #x_tr = RandomRotation(90, resample=False, expand=False, center=None)(x_tr)
        x_tr = transforms.ToTensor()(x_tr)
        
        y_tr = torch.from_numpy(np.array(self.y_tr[idx]))
        idx = torch.from_numpy(np.array(idx))
        return x_tr,y_tr,idx
    
def load_lymphoma(data_path='./lymphoma_jpg'):

    #normalize = transforms.Normalize(mean=[0.5],std=[0.5])     
    
    train_dataset = datasets.ImageFolder(data_path, transforms.Compose([ #Transform only when get_item())
            #transforms.Grayscale(num_output_channels=1), #When using ImageFolder class and with no custom loader, pytorch uses PIL to load image and converts it to RGB
            #transforms.Resize(224),
            #transforms.RandomSizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ])) 
    
    
    imgs_tuples = train_dataset.imgs
    
    data_tensor = None
    target_tensor = []
    for i in range  (374): #374 (): # (len(imgs_tuples)):
        #index = random.randint(1,860)
        index = i
        path,target = imgs_tuples[index]
        image = train_dataset.loader(path)
        
        image = transforms.Resize((224,224))(image)

        #image = transforms.functional.adjust_contrast(image,2)

        #image = transforms.fuctional.adjust_gamma(image,2)
        image = transforms.ToTensor()(image)
        #image = image+0.
        #image = transforms.Normalize(mean=[0.5],std=[0.5])(image)  
        #print('image shape',image.shape)
        if data_tensor is None:
            img = torch.unsqueeze(image, 0)   #For color image of N channels
            data_tensor = img
            target_tensor.append(target)
        else:
            img = torch.unsqueeze(image, 0)      #For color image of N channels      
            data_tensor = torch.cat((data_tensor, img),0)
            target_tensor.append(target)
        #print("image = {} target= {}".format(i,target))
        
    #data_tensor = data_tensor >0.2
    #data_tensor = data_tensor.float()
    #data_tensor = torch.squeeze(data_tensor,1)
    data_train = np.array(data_tensor)
    labels_train =   np.array(target_tensor)
    #print('labels_train = ',labels_train)
    data_train = data_train.astype(np.float32)
    #data_train = np.expand_dims(data_train,axis = 1) #to be processed by convolution layer
    #data_train = np.reshape(data_train,[-1,1,16,16])
    labels_train = labels_train.astype(np.int64)
    

    x = data_train
    #x = np.divide(x, 255.)
    y = labels_train
    print( 'Lymphoma samples', x.shape)
    #print('Hela true labels',train_dataset.class_to_idx)
    return x, y

class LymphomaDataset(Dataset):

    def __init__(self,x_train =None, y_train =None):
        
        if x_train is None and y_train is None:
            x_tr, y_tr= load_lymphoma()
            x_tr = torch.from_numpy(x_tr)
        else:
            x_tr= x_train
            y_tr = y_train
        self.x_tr = x_tr            
        #self.x_tr = resize_tensor(x_tr,28,28)
        #x_tr = torch.nn.functional.interpolate(x_tr,[28,28],mode='bilinear')
        #self.x_tr = np.divide(x_tr, 255.)

        #print('x_tr shape',self.x_tr.shape)
        #print('x 0 values',x_tr[0,0,5:15,5:15])
        print('x min values',torch.min(self.x_tr))
        print('x max values',torch.max(self.x_tr))
        
        self.y_tr = y_tr
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
        #x_tr = torch.from_numpy(np.array(self.x_tr[idx]))
        #x_tr = resize_tensor(x_tr,28,28)
        x_tr = self.x_tr[idx]
        
        x_tr = transforms.ToPILImage()(x_tr)
        
        x_tr =transforms.RandomCrop((224,224), padding=56, padding_mode='symmetric')(x_tr)
        #x_tr = transforms.RandomCrop((224,224))(x_tr)
        
        x_tr = RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3)(x_tr)
        x_tr = RandomHorizontalFlip(p=0.5)(x_tr)
        x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        #x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(x_tr)
        #x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        x_tr = RandomRotation(30, resample=False, expand=False, center=None)(x_tr)
        
        x_tr = transforms.ToTensor()(x_tr)
        
        y_tr = torch.from_numpy(np.array(self.y_tr[idx]))
        idx = torch.from_numpy(np.array(idx))
        return x_tr,y_tr,idx   
    
def load_multidomain(data_path='./Multidomain_dataset'): #(data_path='./dataset_papsmear'): #(data_path='./lymphoma_jpg'): #(data_path='./Hela_jpg'):

    #normalize = transforms.Normalize(mean=[0.5],std=[0.5])     
    
    train_dataset = datasets.ImageFolder(data_path)
        
    imgs_tuples = train_dataset.imgs
    
    data_tensor = None
    target_tensor = []
    for i in range  (2153): #374 + 917 + 862 (): # (len(imgs_tuples)):
        #index = random.randint(1,860)
        index = i
        path,target = imgs_tuples[index]
        image = train_dataset.loader(path)
        
        image = transforms.Resize((224,224))(image)

        #image = transforms.functional.adjust_contrast(image,2)

        #image = transforms.fuctional.adjust_gamma(image,2)
        image = transforms.ToTensor()(image)
        #image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)        
        img = torch.unsqueeze(image, 0)         
        #image = image+0.
        #image = transforms.Normalize(mean=[0.5],std=[0.5])(image)  
        #print('image shape',image.shape)
        if data_tensor is None: 
            data_tensor = img
        else:  
            data_tensor = torch.cat((data_tensor, img),0)
        target_tensor.append(target)
        #print("image = {} target= {}".format(i,target))
        
    x_tr = np.array(data_tensor, dtype=np.float32)
    y_tr =   np.array(target_tensor,dtype=np.int64)
    print( 'training samples', x_tr.shape)
    #print('Hela true labels',train_dataset.class_to_idx)
    return x_tr, y_tr

class MultiDataset(Dataset):
    def __init__(self,x_train =None, y_train =None):
        if x_train is None and y_train is None:
            x_tr, y_tr = load_multidomain()
            x_tr = torch.from_numpy(x_tr)

        else:  #when harmining

            x_tr = x_train
            y_tr = y_train


        self.x_tr = x_tr           
        self.y_tr = y_tr

     
        print('x min values',torch.min(self.x_tr))
        print('x max values',torch.max(self.x_tr))
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
        #x_tr = torch.from_numpy(np.array(self.x_tr[idx]))
        #x_tr = resize_tensor(x_tr,28,28)
        x_tr = self.x_tr[idx]
        
        
        x_tr = transforms.ToPILImage()(x_tr)
        
        x_tr =transforms.RandomCrop((224,224), padding=56, padding_mode='symmetric')(x_tr)
        #x_tr = transforms.RandomCrop((224,224))(x_tr)
        
        x_tr = RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3)(x_tr)
        x_tr = RandomHorizontalFlip(p=0.5)(x_tr)
        x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        #x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(x_tr)
        #x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        x_tr = RandomRotation(30, resample=False, expand=False, center=None)(x_tr)
        
        x_tr = transforms.ToTensor()(x_tr)
        '''
        img_size = (224,224)
        sigma_min  =0.2
        sigma_max = 8.0
        p_min = 0.4
        p_max = 0.6
        x_tr = cow_mask(x_tr, img_size, sigma_min, sigma_max, p_min, p_max)
        '''
        y_tr = torch.from_numpy(np.array(self.y_tr[idx]))
        idx = torch.from_numpy(np.array(idx))
        return x_tr,y_tr,idx        