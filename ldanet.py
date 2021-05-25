# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

from __future__ import print_function, division
import warnings

import argparse
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans

import torch
import torch.nn as nn

from torch.optim import Adam, lr_scheduler

#from torchvision import datasets
from utils_lda import MultiDataset, roc_curve_plot, hard_mine_classes
from earlystopping import EarlyStopping

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

#%matplotlib inline
#from sklearn.decomposition import PCA
from models.imagenet import mobilenetv1, fd_mobilenetv1, mobincep, mobilenetv2

N = 5000 # limit number of samples for scattering show

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123
#########################################################################

class LDAnet(nn.Module):

    def __init__(self,
                 n_classes,
                 pretrain_path= 'pretrained/mobilenetv2_1.0-0c6065bc.pth'):
        super(LDAnet, self).__init__()

        self.pre_path = pretrain_path
        #self.model_conv = mobilenetv2()
        #self.model_conv = fd_mobilenetv1()
        self.model_conv = mobincep()
        #self.model_conv.load_state_dict(torch.load(self.pre_path))



        num_ftrs = self.model_conv.classifier.in_features
        self.model_conv.classifier = nn.Linear(num_ftrs, n_classes)

    def train_model(self,dataset,train_set_loader,nepochs,  split_k, l_r, path=''):

        if not path == '':
            self.load_pretrained(path)

        for param in self.model_conv.parameters():
            param.requires_grad = True

        run_train(self.model_conv,dataset,train_set_loader,nepochs,split_k,l_r)



    def load_pretrained(self, path=''):
        #load pretrain weights
        self.model_conv.load_state_dict(torch.load(path))
        print('load model_conv from', path)
    def forward(self, x):

        y = self.model_conv(x)
        #print("------------",y.shape)
        return y




def run_train(model,dataset,data_set_loader,nepochs,split_k,l_r): #FOR MINIBATCH TRAINING
    '''
    pretrain model
    '''
    #warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta)


    #Supervised learning loss
    #criterion = nn.CrossEntropyLoss().cuda()
    #weights = [0.8, 0.02]
    #class_weights=torch.FloatTensor(weights)
    #criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss().to(device)


    kmeans = KMeans(n_clusters=args.num_classes, n_init=20)
    centroids = None
    _,_,out_score = compute_inference(model,train_set_data,train_set_y)
    _ = kmeans.fit_predict(out_score)
    centroids = kmeans.cluster_centers_

    #optimizer = Adam(model.parameters(), lr = l_r )
    optimizer = Adam(model.parameters(), lr=l_r,eps=1e-5, amsgrad=True ) #BETTER vs ADAM   and SGD
    #optimizer = torch.optim.SGD(model.parameters(), l_r, momentum=0.9,weight_decay=1e-4)
    # Observe that only parameters of final layer are being optimized
    #optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Observe that all parameters are being optimized
    #optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)


    model.cuda()
    model.to(device)
    #hidden = torch.tensor([]).cuda()


    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    model.train()
    #distribution = normal.Normal(0.0,math.sqrt(1))

    #SUPERVISED TRAINING
    for epoch in range(nepochs): #200
        #hidden = torch.tensor([]).cuda()
        #true_target= torch.tensor([],dtype=torch.int64).cuda()
        centroids = torch.tensor(centroids,dtype=torch.float).to(device)
        output_scores = torch.tensor([]).cuda()
        model.train()
        train_loss = 0
        running_corrects = 0.0
        samples_count = 0
        #model.train() vs model.eval() # there is a big difference in prediction accuracy when switch between two modes, due to batch_norm???
        #During training, this layer keeps a running estimate of its computed mean and variance. The running sum is kept with a default momentum of 0.1.
        #During evaluation, this running mean/variance is used for normalization.
        for batch_idx, (x, y,_) in enumerate(data_set_loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            output_scores = torch.cat((output_scores,outputs))
            _, preds = torch.max(outputs, 1)
            #hidden = torch.cat((hidden,z_batch))
            #true_target = torch.cat((true_target,y))
            #print('true_target size =', true_target.size())
            supervised_loss = criterion(outputs,y)
            q_svdd_centroids = (torch.sum(torch.pow(outputs.unsqueeze(1) - centroids, 2), 2) )
            q_svdd,_ = torch.min(q_svdd_centroids, dim=1, keepdim=True)
            svdd_loss = torch.sum(q_svdd,0)/x.shape[0]
            total_d = torch.sum(q_svdd_centroids, dim=1, keepdim=True)
            d_xi_cj = total_d - q_svdd
            d_xi_cj_loss = 1.0/torch.sum(d_xi_cj,0)/x.shape[0]

            loss = 1.0*supervised_loss + args.gamma*svdd_loss + args.gamma2*d_xi_cj_loss #0.1 svdd -> 0.11 acc; 0*svdd ->0.88 acc

            running_corrects += torch.sum(preds == y.data)
            samples_count += len(y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            #exp_lr_scheduler.step()   #scheduler.step() should be called after optimizer.step()
        #acc__ep = (hidden.argmax(-1).detach().cpu().numpy() == true_target.cpu().numpy())
        #print('------------------------------------ non-accurate preds = ', len(acc__ep) - acc__ep.sum() )
        #print('true_target size =', true_target.size())
        '''
        if epoch>200:
            #for params in model.parameters():
                #params.requires_grad = True
            new_lr = args.lr*0.5
            print('new lr =', new_lr)
            optimizer = Adam(model.parameters(), lr=new_lr, eps=1e-5, amsgrad=True)
            #print('lr = ', args.lr)
        '''


        epoch_train_loss.append( train_loss)

        train_acc = running_corrects.double() / samples_count
        epoch_train_acc.append(train_acc)
        #print("epoch {} loss={:.4f}".format(epoch, train_loss))
        #epoch_train_loss_u.append( train_loss_u)
        print("epoch {} Train Loss = {:.4f}  Train Acc = {:.4f}".format(epoch, train_loss, train_acc))


        ############################# CHECK VALIDATION LOSS

        #model.eval()
        if epoch%1 == 0:
            model.eval()
            with torch.no_grad():
                val_loss,val_acc,_ = compute_inference(model,val_set_data,val_set_y)
                epoch_val_loss.append( val_loss)
                epoch_val_acc.append(val_acc)
                print('Check epoch {}: -- Val Loss {:.4f}   Val Acc {:.4f}'.format(epoch,val_loss, val_acc))
                #Update centroids
            _ = kmeans.fit_predict(output_scores.detach().cpu().numpy())
            centroids = kmeans.cluster_centers_


        if epoch > nepochs/2:
            # early_stopping needs the VALIDATION loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, model,args.output_path+'split_k'+str(split_k)+'___'+args.pretrain_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        print('-------------------------------------------------------')


    fig, axs = plt.subplots(2, 2)
    st = fig.suptitle("Curve", fontsize="x-large")
    axs[0, 0].plot(epoch_train_loss,'r')
    axs[0, 0].set_title('Training loss')
    axs[0, 1].plot(epoch_train_acc,'b')
    axs[0, 1].set_title('Training accuracy')
    axs[1, 0].plot(epoch_val_loss,'r')
    axs[1, 0].set_title('Val loss')
    axs[1, 1].plot(epoch_val_acc,'b')
    axs[1, 1].set_title('Val accuracy')

    fig.tight_layout()
    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85, hspace=0.5)
    plt.savefig(args.output_path+'PLOT_TRAINING_CURVES___split_k'+str(split_k)+'___'+'.pdf')
    plt.close()



    # Report the performance of the saved check point
    print(' ------------------------------------------------------ ')
    print(' Report the performance of the saved check point ')
    model_checkpoint = LDAnet(n_classes=args.num_classes).to(device)
    model_checkpoint.load_pretrained(args.output_path+'split_k'+str(split_k)+'___'+args.pretrain_path)
    model_checkpoint.eval()    #Caution of Batch Normalization

    print('------------------------------------------------------')
    testing_loss, testing_acc,out_score = compute_inference(model_checkpoint,testing_set_k_data, testing_set_k_y)

    print('After trained with training set: Testing Acc {:.4f}'.format(testing_acc))

    roc_curve_plot(testing_set_k_y,out_score,args.num_classes,split_k)  #y_true_u should be converted to one-hot vector





    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(out_score)
    fashion_scatter(fashion_tsne_hidden_without_pca, testing_set_k_y ,'Aftertrained___split_k'+str(split_k)+'___')



    #torch.save(model.state_dict(), args.output_path+'split_k'+str(split_k)+'___'+args.pretrain_path)
    #print("model saved to {}.".format(args.output_path+'split_k'+str(split_k)'___'+args.pretrain_path))


def compute_inference(model,data,labels):          #FOR MINIBATCH REFERENCE
    partial_size = args.batch_size
    criterion = nn.CrossEntropyLoss()
    model.eval() #Caution of Batch Normalization
    #out_score = torch.tensor([])
    out_score = np.empty((0,args.num_classes), dtype=float)   #columns should be equal n_classes
    #print('out_score =',out_score)
    #print("data shape = ", data.shape)
    data_size = data.shape[0]
    m = int(data_size/partial_size)
    n = data_size%partial_size
    inference_loss = 0
    inference_diff = 0
    inference_acc = 0
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        partial_labels = labels[i*partial_size:(i+1)*partial_size]
        partial_data = partial_data.to(device)
        #print('partial_data shape =',partial_data.shape)
        out_score_partial = model(partial_data)
        #print('out_score_partial shape = ', out_score_partial.shape)
        inference_loss_partial = criterion(out_score_partial,torch.tensor(partial_labels, dtype=torch.long).to(device))
        inference_loss+=inference_loss_partial.item()
        #print('inference_loss = ',inference_loss)
        diff_partial = (out_score_partial.argmax(-1).detach().cpu().numpy() == np.array(partial_labels))
        diff_partial_sum = diff_partial.sum()
        inference_diff += diff_partial_sum
        #print('out_score_partial shape = ', out_score_partial.shape)
        #print('out_score_partial.data.cpu().numpy() shape = ' , out_score_partial.data.cpu().numpy().shape)
        out_score = np.append(out_score,out_score_partial.data.cpu(),axis=0)

        #torch.cuda.empty_cache()
    if n>0:
        partial_data = data[m*partial_size:]
        partial_labels = labels[m*partial_size:]
        partial_data = partial_data.to(device)
        #print('partial_data shape =',partial_data.shape)
        out_score_partial = model(partial_data)
        #print('out_score_partial shape = ', out_score_partial.shape)
        inference_loss_partial= criterion(out_score_partial,torch.tensor(partial_labels, dtype=torch.long).to(device))
        inference_loss+=inference_loss_partial.item()
        #print('inference_loss = ',inference_loss)
        diff_partial = (out_score_partial.argmax(-1).detach().cpu().numpy() == np.array(partial_labels))
        diff_partial_sum = diff_partial.sum()
        inference_diff += diff_partial_sum
        #print('out_score_partial shape = ', out_score_partial.shape)
        #print('out_score_partial.data.cpu().numpy() shape = ' , out_score_partial.data.cpu().numpy().shape)
        out_score = np.append(out_score,out_score_partial.data.cpu(),axis=0)

        #torch.cuda.empty_cache()
    inference_acc = inference_diff/data_size
    #print('out_score shape =', out_score.shape)
    #print('Finish compute inference ')
    return inference_loss, inference_acc,out_score


def indices_shuffling(dataset):
    from sklearn.model_selection import StratifiedKFold
    X = dataset.x_tr
    y = dataset.y_tr
    #skf = StratifiedKFold(n_splits=5,random_state=random.randint(1,1000),shuffle=True) #corresponding to split  ratio of 0.8
    skf = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    skf.get_n_splits(X, y)

    minor_set_indices = []


    for major_set_index, minor_set_index in skf.split(X, y):
        minor_set_indices.append(minor_set_index) #indices +=unsupervised_index


    return minor_set_indices
def split_part_k_sample(train_dataset,minor_set_indices, k):
    num_subsets = len(minor_set_indices)
    subset_idx = list(range(0,num_subsets))

    testing_set_k_idx = minor_set_indices[k]
    subset_idx.remove(k)
    #Select randomly one minor set for training, and another set for validation
    v = subset_idx[0]
    val_set_idx = minor_set_indices[v]
    subset_idx.remove(v)
    train_set_idx = []
    for t in subset_idx:
      train_set_idx.extend(minor_set_indices[t])

    train_set_sampler = SubsetRandomSampler(train_set_idx)

    train_set_loader = torch.utils.data.DataLoader(
        train_dataset,sampler=train_set_sampler,
        batch_size=args.batch_size,
        num_workers=0, pin_memory=True)


    return train_set_loader,train_set_idx, val_set_idx, testing_set_k_idx





def train_LDAnet(dataset,train_set_loader,split_k):

    #warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


    model = LDAnet(n_classes=args.num_classes).to(device)
    model.train_model(dataset,train_set_loader,args.nepochs,split_k,args.lr,'')
    #We need to load againn the model, otherwise the peformance is decreased
    # because the model after trained (not the checkpoint) saved the current parameters of the batchnorm layers
    model.load_pretrained(args.output_path+'split_k'+str(split_k)+'___'+args.pretrain_path)

    model.eval()    #Caution of Batch Normalization



    print('------------------------------------------------------')
    testing_loss, testing_acc,out_score = compute_inference(model,testing_set_k_data,testing_set_k_y)

    print('Loading pretrained model: Testing Acc {:.4f}'.format(testing_acc))
    '''
    roc_curve_plot(testing_set_k_y,out_score,args.num_classes,split_k)  #y_true_u should be converted to one-hot vector

    print('------------------------------------------------------')

    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(out_score)
    fashion_scatter(fashion_tsne_hidden_without_pca,testing_set_k_y,'Beginning___split_k'+str(split_k)+'___')

    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    '''

def fashion_scatter(x, colors,message):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # show the text for digit corresponding to the true label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0) #true labels
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    #plt.savefig('./results_vae/scatter_'+ str(idx) + '.png', bbox_inches='tight')
    plt.savefig(args.output_path+'scatter_'+ '_'+message+ '.pdf', bbox_inches='tight')
    plt.close()
    return f, ax, sc, txts



if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split_k', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_classes', default=15, type=int)
    parser.add_argument('--nepochs', default=5001, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--pretrain_path', type=str, default='multidataset.pkl')

    parser.add_argument('--gamma', default=0.001, type=float, help='coefficient of clustering loss') #0.001
    parser.add_argument('--gamma2', default=0.1, type=float, help='coefficient 2 of clustering loss') #0.1

    parser.add_argument('--output_path', type=str, default='./results/')
    # early stopping patience; how long to wait after last time validation loss improved.
    parser.add_argument('--patience', default=2500, type=int)
    #Minimum change in the monitored quantity to qualify as an improvement.
    parser.add_argument('--delta', default=0, type=int)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    #print("use cuda: {}".format(args.cuda))
    print(args)
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = MultiDataset()
    minor_set_indices = indices_shuffling(dataset)
    #r20
    train_set_loader,train_set_idx, val_set_idx, testing_set_k_idx = split_part_k_sample(dataset,minor_set_indices, k=args.split_k)
    data = dataset.x_tr
    labels = dataset.y_tr

    train_set_data = [data[i] for i in train_set_idx]
    train_set_data = torch.stack(train_set_data)#.to(device)
    train_set_y = [labels[i] for i in train_set_idx]
    train_set_y = np.array(train_set_y)

    testing_set_k_data = [data[i] for i in testing_set_k_idx]
    testing_set_k_data = torch.stack(testing_set_k_data)#.to(device)
    testing_set_k__y = [labels[i] for i in testing_set_k_idx]
    testing_set_k_y = np.array(testing_set_k__y)

    val_set_data = [data[i] for i in val_set_idx]
    val_set_data = torch.stack(val_set_data)#.to(device)
    val_set__y = [labels[i] for i in val_set_idx]
    val_set_y = np.array(val_set__y)



    train_LDAnet(dataset,train_set_loader, split_k=args.split_k)



