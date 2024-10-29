import torch
import torch.utils.data.dataloader
import torchvision
import numpy
import matplotlib.pyplot as plt
import torchvision.transforms as tr
import numpy as np
import os

#models

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def train_nn(model,train_loader,test_loader,crieterion,optimizer,epocs):
    device=set_device()
    best=0
    for epoc in range(epocs):
        print("epoc no %d" %(epoc+1))
        model.train()
        running_loss=0.0
        running_correct=0.0
        total=0
        for data in train_loader:
            images,labels=data #32 IMAGES AND LABELS
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)

            optimizer.zero_grad()

            outputs=model(images) #outputs has predictions

            _,predicted=torch.max(outputs.data,1) #INPUT AND DIMENTIONS(1) COMPUTE MAX OF A TENSOR IF LARGE TENSOR THEN LIST OF MAX TENSOR(index is returned)
            loss=crieterion(outputs,labels) #predicts loss between outputs predicteed and labels

            loss.backward() #back propagate to know how much each parameter should change to reduce the loss.
            optimizer.step() #applies the update rule defined by the optimization algorithm to each parameter

            running_loss+=loss.item() #lost image count

            running_correct+=(labels==predicted).sum().item()
        epoch_loss=running_loss/len(train_loader)
        epoch_accuracy=100.00*running_correct/total

        print("Training dataset got %d out og %d images correctly (%.3f%%).Epoch loss: %.3f"%(running_correct,total,epoch_accuracy,epoch_loss))
        
        val=evaluate_model_on_test(model,test_loader)
        if(best<val):
            save_checkpoint(model,epoc,optimizer,best)
        
    print("Finished")
    return model


def evaluate_model_on_test(model,test_loader):
    model.eval()
    predicted_correctly_onepoch=0
    total=0
    device=set_device()
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)

            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)

            predicted_correctly_onepoch+=(labels==predicted).sum().item()
    epoc_acc=100.00*predicted_correctly_onepoch/total

    print("Testing Dataset Got %d out of %d images correctly found (%.3f%%)"
          %(predicted_correctly_onepoch,total,epoc_acc))       
    return epoc_acc

def save_checkpoint(model,epoc,optimizer,best):
    state={
        "model":model.state_dict(),
        "epoch":epoc+1,
        "best_accuracy":best,
        "optimizer":optimizer
    }
    torch.save(state,"best_model.tar")
# mean=[1,2,3]
# std=[1,2,3]

train_path="/home/tst/Desktop/Python/train"
test_path="/home/tst/Desktop/Python/test"

train_transform=tr.Compose([
    tr.Resize((224,224)),
    tr.RandomHorizontalFlip(),
    tr.RandomRotation(10),
    tr.ToTensor(),
    #tr.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

test_transform=tr.Compose([
    tr.Resize((224,224)),
    tr.RandomHorizontalFlip(),
    tr.RandomRotation(10),
    tr.ToTensor(),
    #tr.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

train_dataset=torchvision.datasets.ImageFolder(root=train_path,transform=train_transform)
test_dataset=torchvision.datasets.ImageFolder(root=test_path,transform=test_transform)

# print(train_dataset.classes)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

def set_device():
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    return torch.device(dev) 

resnet18_model=models.resnet18(weights=None)

num_fts=resnet18_model.fc.in_features
number_of_classes=10

resnet18_model.fc=nn.Linear(num_fts,number_of_classes)
device=set_device()

resnet18_model=resnet18_model.to(device)

loss_fn=nn.CrossEntropyLoss()

#lr=learning rate
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_nn(resnet18_model,train_loader,test_loader,loss_fn,optimizer,15)

#can also be used


#checkpoint=torch.load("best_model.tar")
#checkpoint = torch.load("best_model.tar", weights_only=True)
# resnet18_model=models.resnet18(weights=None)

# num_fts=resnet18_model.fc.in_features
# number_of_classes=10
# resnet18_model.fc=nn.Linear(num_fts,number_of_classes)

# resnet18_model.load_state_dict(checkpoint["model"])

# torch.save(resnet18_model,"Model.pth")


# Save the foldername of each dataset
def get_file_names_from_directory(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    file_names.sort()
    return file_names
filenames=get_file_names_from_directory(train_path)
np.savez_compressed("classes.npz",filenames)



checkpoint = torch.load("best_model.tar",weights_only=False)
resnet18_model.load_state_dict(checkpoint["model"])
torch.save(resnet18_model, "Model.pth")