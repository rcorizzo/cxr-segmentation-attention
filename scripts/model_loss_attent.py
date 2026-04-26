#!/usr/bin/env python
# coding: utf-8

# Customize loss function: loss = loss_class+loss_attention, loss_attention=(1-ratio)*gtLabel

# In[ ]:


import os
import numpy as np
import torch
import glob
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
import pathlib
import timm
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
import copy
import sys
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
import time
import argparse
from skimage.draw import polygon
from model import build_unet


# In[ ]:


parser = argparse.ArgumentParser(description='Script for training and evaluating different models with the proposed loss')
parser.add_argument('--dataset',type=str, required=True, help="Name of data: ottawa or nih")
parser.add_argument('--path',type=str, required=True, help="Path of data")
parser.add_argument('--backbone',type=str, required=True, help="Type of model: pvt, vgg or resent")
parser.add_argument('--task',type=str, required=True, help="Specify a classification task: ne, np, nep or neps")
parser.add_argument('--gpu',type=int, default= 0, help = 'Specify a gpu if there are more than one')
parser.add_argument('--batch',type=int, default= 32, help = 'Specify the batch size')
parser.add_argument('--lamda',type=float, default=None, help = 'Specify a lambda value betweeen 0 to 1')
parser.add_argument('--thresh',type=float, default= 0.3, help = 'Specify a threshold value betweeen 0 to 1')
parser.add_argument('--isAdaptive', action='store_true', help='A boolean flag indicates whether the value of lambda is adaptive')

args = parser.parse_args()


# In[ ]:


start_time = time.time() 
torch.cuda.set_device(args.gpu)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


class MyDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(MyDataset, self).__init__(root, transform)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __getitem__(self, idx):
        image, label = super(MyDataset, self).__getitem__(idx)
        path = self.imgs[idx][0]  
        
        return image, label, path


# In[ ]:


transformer=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                            [0.5,0.5,0.5])
    ])

train_path = args.path+'train'
valid_path = args.path+'val'
test_path = args.path+'test'

train_dataset = MyDataset(root=train_path, transform=transformer)
valid_dataset = MyDataset(root= valid_path, transform=transformer)
test_dataset = MyDataset(root=test_path, transform=transformer)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.png'))
valid_count=len(glob.glob(valid_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))
print(train_count,valid_count,test_count)

checkpoint_path = "../model/unet_model.pt"
# unet_model = torch.load(checkpoint_path)

# Load unet model to cpu followed by loading to another gpu
unet_model = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
unet_model = unet_model.to(device)
unet_model.eval()

def box(img_path):
    H = 512
    W = 512
    size = (W, H)
    
#     name = img_path.split("/")[-1].split(".")[0]
    x_unet = cv2.imread(img_path,cv2.IMREAD_COLOR)
    x_unet = cv2.resize(x_unet,size)
    x_unet = np.transpose(x_unet, (2, 0, 1))
    x_unet = x_unet/255.0
    x_unet = np.expand_dims(x_unet, axis=0)
    x_unet = torch.from_numpy(x_unet).float() 
    x_unet = x_unet.to(device)
    
    with torch.no_grad(): # Stop calculate gradients
        """ Prediction"""
        pred_y = unet_model(x_unet)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512), 1 represents batch size, not channel, transform to cpu
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512), remove 1st dimension(removing batch)
        pred_y = pred_y > 0.5 # Convert to 0 and 1
        pred_y = np.array(pred_y, dtype=np.uint8)
        
    model_size=(224,224)
    pred_y = cv2.resize(pred_y.astype(np.uint8), model_size)
    contours, hierarchy = cv2.findContours(pred_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs,ys,ws,hs=[],[],[],[]
    
    for contour in contours:
        X,Y,W,H = cv2.boundingRect(contour)
        xs.append(X)
        ys.append(Y)
        ws.append(X+W)
        hs.append(Y+H)

    if len(contours) != 0:
        box_x=min(xs)
        box_y=min(ys)
        box_w=max(ws)-box_x
        box_h=max(hs)-box_y

        box_x=round(box_x)
        box_y=round(box_y)
        box_w=round(box_w)
        box_h=round(box_h)

        return box_x, box_y, box_w, box_h
    
    else:
        return 0, 0, 224, 224


# In[ ]:


# Define backbone model
if args.task in ['ne','np','ns']:
    num_class=2
elif args.task == 'nep':
    num_class=3
else:
    num_class=4

if args.backbone == 'pvt':
    model = timm.create_model('pvt_v2_b2_li', pretrained=True, num_classes=num_class).to(device)
elif args.backbone == 'vgg':
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_class)
    model.to(device)
else:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, num_class)
    model.to(device)


# In[39]:


# Loss function for Effusion
class CustomLoss_ne(nn.Module):
    def __init__(self):
        super(CustomLoss_ne, self).__init__()

    def forward(self, images, outputs, labels, predictions, paths):
        losses = torch.zeros(images.size(0))  # Build a tensor of the same size as the batch to store the loss values of each image
#         model = backbone(device)
        model2 = copy.deepcopy(model)

        with torch.no_grad():
            if args.backbone == 'pvt':
                target_layers = [model2.stages[-1]]
            elif args.backbone == 'vgg':
                target_layers = [model2.features[28]]
            else:
                target_layers = [model2.layer4[-1]]
            
            use_cuda = torch.cuda.is_available()
            cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=use_cuda)

        for i in range(images.size(0)):
            loss_ce = F.cross_entropy(outputs[i], labels[i])

            image_tensor = images[i].unsqueeze(0)
            predict = predictions[i].item()

            targets = [ClassifierOutputTarget(predict)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

#             threshold = args.thresh
            box_x, box_y, box_w, box_h = box(paths[i])
            target_area = grayscale_cam[round(box_y+(box_h)/2):box_y+box_h, box_x:box_x+box_w] # This is new bottom area that calculated from current mask
            target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
            red_pixels_target = torch.sum(target_area_tensor > threshold) 
            total_pixels_target = target_area_tensor.numel()
            red_area_ratio_target = red_pixels_target / total_pixels_target
            loss_attent = (1 - red_area_ratio_target) * labels[i].item()

#             lamda=args.lamda
            loss_combined = lamda*loss_ce + (1-lamda)*loss_attent
            losses[i] =  loss_combined

        return torch.mean(losses)


# In[ ]:


# Loss function for Pneumothorax
class CustomLoss_np(nn.Module):
    def __init__(self):
        super(CustomLoss_np, self).__init__()

    def forward(self, images, outputs, labels, predictions, paths):
        losses = torch.zeros(images.size(0))  # Build a tensor of the same size as the batch to store the loss values of each image
#         model = backbone(device)
        model2 = copy.deepcopy(model)

        with torch.no_grad():
            if args.backbone == 'pvt':
                target_layers = [model2.stages[-1]]
            elif args.backbone == 'vgg':
                target_layers = [model2.features[28]]
            else:
                target_layers = [model2.layer4[-1]]
            
            use_cuda = torch.cuda.is_available()
            cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=use_cuda)

        for i in range(images.size(0)):
            loss_ce = F.cross_entropy(outputs[i], labels[i])

            image_tensor = images[i].unsqueeze(0)
            predict = predictions[i].item()

            targets = [ClassifierOutputTarget(predict)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

#             threshold = args.thresh
            box_x, box_y, box_w, box_h = box(paths[i])
            target_area = grayscale_cam[box_y:round(box_y+(box_h)/2), box_x:box_x+box_w] # This is new top area that calculated from current mask
            target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
            red_pixels_target = torch.sum(target_area_tensor > threshold) 
            total_pixels_target = target_area_tensor.numel()
            red_area_ratio_target = red_pixels_target / total_pixels_target
            loss_attent = (1 - red_area_ratio_target) * labels[i].item()

#             lamda=args.lamda
            loss_combined = lamda*loss_ce + (1-lamda)*loss_attent
            losses[i] =  loss_combined

        return torch.mean(losses)


# In[ ]:


# Loss function for SubcuEmphysema
class CustomLoss_ns(nn.Module):
    def __init__(self):
        super(CustomLoss_ns, self).__init__()

    def forward(self, images, outputs, labels, predictions, paths):
        losses = torch.zeros(images.size(0))  # Build a tensor of the same size as the batch to store the loss values of each image
#         model = backbone(device)
        model2 = copy.deepcopy(model)

        with torch.no_grad():
            if args.backbone == 'pvt':
                target_layers = [model2.stages[-1]]
            elif args.backbone == 'vgg':
                target_layers = [model2.features[28]]
            else:
                target_layers = [model2.layer4[-1]]
            
            use_cuda = torch.cuda.is_available()
            cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=use_cuda)

        for i in range(images.size(0)):
            loss_ce = F.cross_entropy(outputs[i], labels[i])

            image_tensor = images[i].unsqueeze(0)
            predict = predictions[i].item()

            targets = [ClassifierOutputTarget(predict)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :] 

#             threshold = args.thresh
            box_x, box_y, box_w, box_h = box(paths[i])
    
            image_test = np.ones((224, 224), dtype=np.uint8) * 255
            image_test[box_y:224, box_x:box_x+box_w] = 0 # Set non-target area (inside rectangular) as 0
            rr_white, cc_white = np.where(image_test == 255) # Set outside target area as 255
            red_pixels_target=sum(grayscale_cam[rr_white, cc_white] > threshold)
            total_pixels_target = len(rr_white)
            red_area_ratio_target = red_pixels_target / total_pixels_target
            loss_attent = (1 - red_area_ratio_target) * labels[i].item()

#             lamda=args.lamda
            loss_combined = lamda*loss_ce + (1-lamda)*loss_attent
            losses[i] =  loss_combined

        return torch.mean(losses)


# In[ ]:


# Loss function for multi-classes: nep
class CustomLoss_nep(nn.Module):
    def __init__(self):
        super(CustomLoss_nep, self).__init__()

    def forward(self, images, outputs, labels, predictions, paths):
        losses = torch.zeros(images.size(0))  # Build a tensor of the same size as the batch to store the loss values of each image
#         model = backbone(device)
        model2 = copy.deepcopy(model)

        with torch.no_grad():
            if args.backbone == 'pvt':
                target_layers = [model2.stages[-1]]
            elif args.backbone == 'vgg':
                target_layers = [model2.features[28]]
            else:
                target_layers = [model2.layer4[-1]]
            
            use_cuda = torch.cuda.is_available()
            cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=use_cuda)

        for i in range(images.size(0)):
            loss_ce = F.cross_entropy(outputs[i], labels[i])

            image_tensor = images[i].unsqueeze(0)
            predict = predictions[i].item()

            targets = [ClassifierOutputTarget(predict)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :] 
#             threshold = args.thresh
            box_x, box_y, box_w, box_h = box(paths[i])
    
            if labels[i].item() == 0: # GT is No finding
                loss_attent = 0
                
            elif labels[i].item() == 1: # GT is Effusion
                target_area = grayscale_cam[round(box_y+(box_h)/2):box_y+box_h, box_x:box_x+box_w] # This is new bottom area that calculated from current mask
                target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
                red_pixels_target = torch.sum(target_area_tensor > threshold) 
                total_pixels_target = target_area_tensor.numel()
                red_area_ratio_target = red_pixels_target / total_pixels_target
                loss_attent = 1 - red_area_ratio_target
                
            elif labels[i].item() == 2: # GT is Pneumothorax
                target_area = grayscale_cam[box_y:round(box_y+(box_h)/2), box_x:box_x+box_w] # This is new top area that calculated from current mask
                target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
                red_pixels_target = torch.sum(target_area_tensor > threshold) 
                total_pixels_target = target_area_tensor.numel()
                red_area_ratio_target = red_pixels_target / total_pixels_target
                loss_attent = 1 - red_area_ratio_target

#             lamda=args.lamda
            loss_combined = lamda*loss_ce + (1-lamda)*loss_attent
            losses[i] =  loss_combined

        return torch.mean(losses)


# In[ ]:


# Loss function for multi-classes
class CustomLoss_neps(nn.Module):
    def __init__(self):
        super(CustomLoss_neps, self).__init__()

    def forward(self, images, outputs, labels, predictions, paths):
        losses = torch.zeros(images.size(0))  # Build a tensor of the same size as the batch to store the loss values of each image
#         model = backbone(device)
        model2 = copy.deepcopy(model)

        with torch.no_grad():
            if args.backbone == 'pvt':
                target_layers = [model2.stages[-1]]
            elif args.backbone == 'vgg':
                target_layers = [model2.features[28]]
            else:
                target_layers = [model2.layer4[-1]]
            
            use_cuda = torch.cuda.is_available()
            cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=use_cuda)

        for i in range(images.size(0)):
            loss_ce = F.cross_entropy(outputs[i], labels[i])

            image_tensor = images[i].unsqueeze(0)
            predict = predictions[i].item()

            targets = [ClassifierOutputTarget(predict)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :] 
#             threshold = args.thresh
            box_x, box_y, box_w, box_h = box(paths[i])
    
            if labels[i].item() == 0: # GT is No finding
                loss_attent = 0
                
            elif labels[i].item() == 1: # GT is Effusion
                target_area = grayscale_cam[round(box_y+(box_h)/2):box_y+box_h, box_x:box_x+box_w] # This is new bottom area that calculated from current mask
                target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
                red_pixels_target = torch.sum(target_area_tensor > threshold) 
                total_pixels_target = target_area_tensor.numel()
                red_area_ratio_target = red_pixels_target / total_pixels_target
                loss_attent = 1 - red_area_ratio_target
                
            elif labels[i].item() == 2: # GT is Pneumothorax
                target_area = grayscale_cam[box_y:round(box_y+(box_h)/2), box_x:box_x+box_w] # This is new top area that calculated from current mask
                target_area_tensor = torch.from_numpy(target_area)  # Converting numpy arrays to PyTorch tensor
                red_pixels_target = torch.sum(target_area_tensor > threshold) 
                total_pixels_target = target_area_tensor.numel()
                red_area_ratio_target = red_pixels_target / total_pixels_target
                loss_attent = 1 - red_area_ratio_target
                
            elif labels[i].item() == 3: # GT is 3 which is SubcuEmphysema
                image_test = np.ones((224, 224), dtype=np.uint8) * 255
                image_test[box_y:224, box_x:box_x+box_w] = 0 # Set non-target area (inside rectangular) as 0
                rr_white, cc_white = np.where(image_test == 255) # Set outside target area as 255
                red_pixels_target=sum(grayscale_cam[rr_white, cc_white] > threshold)
                total_pixels_target = len(rr_white)
                red_area_ratio_target = red_pixels_target / total_pixels_target
                loss_attent = 1 - red_area_ratio_target

#             lamda=args.lamda
            loss_combined = lamda*loss_ce + (1-lamda)*loss_attent
            losses[i] =  loss_combined

        return torch.mean(losses)


# In[40]:


#Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=1e-10,verbose=True)

if args.task=='ne':
    loss_function = CustomLoss_ne()
elif args.task=='np':
    loss_function = CustomLoss_np()
elif args.task=='ns':
    loss_function = CustomLoss_ns()
elif args.task=='nep':
    loss_function = CustomLoss_nep()
else:
    loss_function = CustomLoss_neps()


# In[ ]:


num_epochs=100
best_accuracy=0.0
no_improvement_epochs = 0

adaptive_suffix = "a" if args.isAdaptive else ""
lamda_suffix = f"{args.lamda}" if args.lamda is not None else ""

model_filename=f"{args.backbone}_{args.dataset}_{args.task}_la{lamda_suffix}{adaptive_suffix}_t{args.thresh}"
model_path=f"results/{model_filename}.pt"

# Variables for plotting
train_losses_plt,train_acces_plt,val_losses_plt,val_acces_plt=[],[],[],[]

for epoch in range(num_epochs):

    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    threshold=args.thresh
    if not args.isAdaptive:
        lamda=args.lamda
    else:
        lamda=1-epoch*0.023
        if lamda<0.2:
            lamda=0.2

    for j, (images, labels, paths) in enumerate(train_loader):
      if torch.cuda.is_available():
          images=Variable(images.cuda())
          labels=Variable(labels.cuda())

      optimizer.zero_grad()

      outputs = model(images)
      _, prediction = torch.max(outputs.data, 1)

      loss = loss_function(images, outputs, labels, prediction, paths)
      loss.backward()
      optimizer.step()

      # train_loss += loss.cpu().data.item()
      train_loss+= loss.cpu().data*images.size(0)
      train_accuracy+=int(torch.sum(prediction==labels.data))

    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count

    # Evaluation on validation set
    model.eval()
    test_loss=0.0
    test_accuracy=0.0
    for j, (images,labels,paths) in enumerate(valid_loader):
        if torch.cuda.is_available():
          images=Variable(images.cuda())
          labels=Variable(labels.cuda())

        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        loss = loss_function(images, outputs, labels, prediction, paths)
        test_loss+= loss.cpu().data*images.size(0)
        # test_loss += loss.cpu().data.item()
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy=test_accuracy/valid_count
    test_loss=test_loss/valid_count

    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test loss: '+str(test_loss)+' Test Accuracy: '+str(test_accuracy))

    # For plotting
    train_losses_plt.append(train_loss)
    train_acces_plt.append(train_accuracy)
    val_losses_plt.append(test_loss)
    val_acces_plt.append(test_accuracy)

     #Save the best model
    if test_accuracy>best_accuracy:
      #torch.save(model.state_dict(),'files/best_checkpoint.model')
      torch.save(model, model_path)
      best_accuracy=test_accuracy
    else:
      no_improvement_epochs+=1

    if no_improvement_epochs>=30:
      print('No improvement for specific epochs. Stopping the training.')
      break

    lr_scheduler.step(test_loss)


# In[ ]:


# Record the end time
end_time = time.time() 
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60

model = torch.load(model_path)
model.eval()

# Evaluate on validation set
y_true,y_pred, y_pred_probabs=[],[],[]

for i, (images,labels,paths) in enumerate(valid_loader):
  if torch.cuda.is_available():
    images=Variable(images.cuda())
    labels=Variable(labels.cuda())

  outputs=model(images)
  _,prediction=torch.max(outputs.data,1)

  y_true.extend(labels.tolist())
  y_pred.extend(prediction.tolist())
  y_pred_probabs.extend(outputs.tolist())

print(classification_report(y_true,y_pred, digits=3))

# Evaluate on unseen test set
y_true2,y_pred2, y_pred_probabs2=[],[],[]

for i, (images,labels,paths) in enumerate(test_loader):
  if torch.cuda.is_available():
    images=Variable(images.cuda())
    labels=Variable(labels.cuda())

  outputs=model(images)
  _,prediction=torch.max(outputs.data,1)

  y_true2.extend(labels.tolist())
  y_pred2.extend(prediction.tolist())
  y_pred_probabs2.extend(outputs.tolist())

print(classification_report(y_true2,y_pred2, digits=3))


# **Specificity, FDR, FOR**

# In[ ]:


def calculate_specificity_fdr_for(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    true_positive = cm[1, 1]
    
    specificity = true_negative / (true_negative + false_positive)
    fdr = false_positive / (false_positive + true_positive)
    for_ = false_negative / (false_negative + true_negative)
    
    return specificity, fdr, for_

if args.task in ['ne','np','ns']:
    
    specificity1, fdr1, for_1=calculate_specificity_fdr_for(y_true,y_pred)
    specificity2, fdr2, for_2=calculate_specificity_fdr_for(y_true2,y_pred2)
    
    y_pred_probabs=np.array(y_pred_probabs)
    softmax_preds = softmax(y_pred_probabs, axis=1)
    positive_probabilities = softmax_preds[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, positive_probabilities)

    # Find the index of the threshold that maximizes the Youden's J statistic
    best_threshold_index = np.argmax(tpr - fpr)

    # Retrieve the best threshold and corresponding FPR and TPR
    best_threshold = thresholds[best_threshold_index]
    best_fpr = fpr[best_threshold_index]
    best_tpr = tpr[best_threshold_index]

    # Calculate AUC
    auc = roc_auc_score(y_true, positive_probabilities)

    # Another unseen test data
    y_pred_probabs2=np.array(y_pred_probabs2)
    softmax_preds2 = softmax(y_pred_probabs2, axis=1)
    positive_probabilities2 = softmax_preds2[:, 1]

    fpr2, tpr2, thresholds2 = roc_curve(y_true2, positive_probabilities2)

    # Find the index of the threshold that maximizes the Youden's J statistic
    best_threshold_index2 = np.argmax(tpr2 - fpr2)

    # Retrieve the best threshold and corresponding FPR and TPR
    best_threshold2 = thresholds2[best_threshold_index2]
    best_fpr2 = fpr2[best_threshold_index2]
    best_tpr2 = tpr2[best_threshold_index2]

    # Calculate AUC
    auc2 = roc_auc_score(y_true2, positive_probabilities2)
    
    original_stdout = sys.stdout

    with open(f"results/{model_filename}.txt", "w") as f:
        sys.stdout = f
        print("Execution Time:", execution_time_minutes, "minutes")
        print(classification_report(y_true,y_pred, digits=3))
        print("AUC Score:", auc)
        print("Best Threshold:", best_threshold)
        print('Specificity: ', specificity1)
        print('False Discovery Rate (FDR): ', fdr1)
        print('False Omission Rate (FOR): ', for_1)
        print('-'*30)
        print("Metrics of unseen test data: ")
        print(classification_report(y_true2,y_pred2, digits=3))
        print("AUC Score:", auc2)
        print("Best Threshold:", best_threshold2)
        print('Specificity: ', specificity2)
        print('False Discovery Rate (FDR): ', fdr2)
        print('False Omission Rate (FOR): ', for_2)
    sys.stdout = original_stdout

else:
    y_true = np.array(y_true)
    y_pred_probabs = np.array(y_pred_probabs)
    y_pred_probabs_softmax = torch.softmax(torch.tensor(y_pred_probabs), dim=1)
    y_pred_probabs_softmax = y_pred_probabs_softmax.numpy()
    auc_val = roc_auc_score(y_true, y_pred_probabs_softmax, multi_class='ovr')

    y_true2 = np.array(y_true2)
    y_pred_probabs2 = np.array(y_pred_probabs2)
    y_pred_probabs_softmax2 = torch.softmax(torch.tensor(y_pred_probabs2), dim=1)
    y_pred_probabs_softmax2 = y_pred_probabs_softmax2.numpy()
    auc_test = roc_auc_score(y_true2, y_pred_probabs_softmax2, multi_class='ovr')
    
    # Generate log file
    original_stdout = sys.stdout
    with open(f"results/{model_filename}.txt", "w") as f:
        sys.stdout = f
        print("Execution Time:", execution_time_minutes, "minutes")
        print(classification_report(y_true,y_pred, digits=3))
        print("AUC Score:", auc_val)
        print('-'*30)
        print("Metrics of unseen test data: ")
        print(classification_report(y_true2,y_pred2, digits=3))
        print("AUC Score:", auc_test)
    sys.stdout = original_stdout
