import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np


def accuracy(preds, labels):
    _, pred_classes = torch.max(preds, 1)
    correct = (pred_classes == labels).float()
    acc = correct.sum() / len(correct)
    return acc
def evaluate(model,valid_dl,loss_func, device='cpu'):
    model.eval()
    batch_losses, batch_accs=[],[]
    for images,labels in valid_dl:
        images, labels = images.to(device), labels.to(device)
        predicted=model(images)
        if type(predicted)!= torch.Tensor : predicted = predicted[0] #In case of PointNet, output is a tuple

        batch_losses.append(loss_func(predicted,labels))
        batch_accs.append(accuracy(predicted,labels))
    epoch_avg_loss=torch.stack(batch_losses).mean().item()# To keep only the mean
    epoch_avg_acc=torch.stack(batch_accs).mean().item()
    return epoch_avg_loss,epoch_avg_acc
def train(model,train_dl,valid_dl,epochs, max_lr, loss_func,optim, device  = "cpu",scheduler_lr = "True", SAVE = True):

    #Choice of the optimization function, Weight decay is associated to L2 Norm
    if optim == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), max_lr,weight_decay=1e-5)
    elif optim == "SGD":
      optimizer= torch.optim.SGD(model.parameters(), max_lr,weight_decay=1e-4)
    elif optim == "Momentum":
      optimizer=torch.optim.SGD(model.parameters(), max_lr,momentum=0.9,weight_decay=1e-4)

    #Adding a learning rate scheduler
    if scheduler_lr :
      scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,epochs*len(train_dl)) #Allow a moving lr, best for fast convergence.


    results=[]
    best_epoch, best_epoch_nb = -1,-1 #To save the best epoch weight.
    #using tqdm for the bouclz over epoch

    for epoch in tqdm(range(1,epochs+1),desc="training in progress"):
    #for epoch in range(1,epochs+1):
        model.train()
        train_losses, train_batch_accs, lrs =[],[],[]

        #Compute over a minibatch
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #Gradient set to zero to avoid accumulation during training (Backprog at the scale of mini-batch not over all the dataset)
            predicted=model(images)
            if type(predicted)!= torch.Tensor : predicted = predicted[0] #In case of PointNet, output is a tuple

            loss=loss_func(predicted,labels)
            train_losses.append(loss)
            loss.backward() #Compute backpropagation
            optimizer.step() # Update weights
    # keep track of learning rate and metrics (here as the dataset is balanced, accuracy alone is relevant)
     
            train_batch_accs.append(accuracy(predicted,labels))
          
            lrs.append(optimizer.param_groups[0]['lr'])
            if scheduler_lr :
                scheduler.step() #Update learning rate


        epoch_train_acc=torch.stack(train_batch_accs).mean().item()
        epoch_train_loss=torch.stack(train_losses).mean().item()
        epoch_avg_loss,epoch_avg_acc = evaluate(model,valid_dl,loss_func) #having the avg metrics for the epoch.

        results.append({'avg_valid_loss': epoch_avg_loss,
                        'avg_val_acc': epoch_avg_acc,
                        'avg_train_loss':epoch_train_loss,
                        'avg_train_acc':epoch_train_acc,
                        'lrs':lrs})

        print('Number of epochs:', epoch,'|',
              'Validation loss :',epoch_avg_loss, ' |','Training loss :'
              ,epoch_train_loss,' |  '
              ,'Training accuracy:', epoch_train_acc
              , 'validation accuracy :',epoch_avg_acc)
        #Save the best model
        
        if SAVE:
          if best_epoch == -1 or epoch_avg_acc > best_epoch:
            best_epoch = epoch_avg_acc
            torch.save(model.state_dict(), f'model_optim_{optim}_epoch_{epoch}_weights.pth')
            if best_epoch_nb != -1 :
              #remove old weight
              os.remove(f'model_optim_{optim}_epoch_{best_epoch_nb}_weights.pth')
            best_epoch_nb = epoch
    #Save the last epoch weight
    if SAVE:
        torch.save(model.state_dict(), f'model_optim_{optim}_epoch_{epoch}_weights.pth')
    return results


################## MODELS ##################
#Model Convolutional layer,lightweight and efficient.
class convnet(nn.Module):
    # Constructor
    def __init__(self,nb_class):
        super(convnet, self).__init__()
        '''
         Convolutional layers
         Conv2d (input channels, output channels, kernel_size, padding)
        '''
        self.nb_class = nb_class

        self.conv_layer_1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16,kernel_size= 3,stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer_2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size= 3,stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer_3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size= 3,stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size= 3,stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
          )

        # Fully Connected layers
        self.hidden_layer = nn.Linear(128*8*8, 256)
        self.output_layer = nn.Linear(256, self.nb_class) #Size to modify.

        #Dropout to avoid over-fitting :
        self.dropout_layer = nn.Dropout(p=0.2) #

    def forward(self, ip):

        # Calling all the convolutional layers
        output = self.conv_layer_1(ip) #Img 128*128 => 64*64
        output = self.conv_layer_2(output) #Img 64*64 => 32*32
        output = self.conv_layer_3(output) #Img 32*32 => 16*16
        output = self.conv_layer_4(output) #IMG 16*16 => 8*8

        # Flattening
        output = torch.flatten(output, 1) #Flatten all dimensions except batch

        # Call fully connected layer
        output = self.hidden_layer(output)

        output = self.dropout_layer(output) #Dropout to avoid over-fitting, dropout will be desactivated in evaluation mode.

        output=self.output_layer(output)

        return output