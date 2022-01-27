import sys

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("/root/deeplearning/model")
from deeplabv3 import DeepLabV3
from unet import UNet
from segnet import SegNet
from AttentionR2UNet import R2AttU_Net
from attentionUnet import AttU_Net
from effunet import EffUNet #Not ready
from unetpp import UnetPlusPlus

sys.path.append("/root/deeplearning/utils")
from utils import add_weight_decay
from mixed_focal_loss import MixedFocalLoss
from focal_loss import FocalLoss
from GWDL import GeneralizedWassersteinDiceLoss

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time


# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "1"

num_epochs = 5000
batch_size = 4
learning_rate = 0.0001

# NOTE! NOTE! chose the architecture to train:
#network = AttU_Net(model_id, project_dir="/root/deeplearning").cuda()
#network = DeepLabV3(model_id, project_dir="/root/deeplearning").cuda()
#network = UNet(model_id = model_id, project_dir="/root/deeplearning").cuda()
#network = SegNet(model_id, project_dir="/root/deeplearning").cuda()
#network = R2AttU_Net(model_id, project_dir="/root/deeplearning").cuda()
#network = EffUNet(model_id, project_dir="/root/deeplearning").cuda() #Not ready
network = UnetPlusPlus(model_id, project_dir="/root/deeplearning").cuda()

train_dataset = DatasetTrain(cityscapes_data_path="/root/deeplearning/data/jetfire",
                            cityscapes_meta_path="/root/deeplearning/data/jetfire/meta")
val_dataset = DatasetVal(cityscapes_data_path="/root/deeplearning/data/jetfire",
                        cityscapes_meta_path="/root/deeplearning/data/jetfire/meta")

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=1)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=1)

params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

with open("/root/deeplearning/data/jetfire/meta/class_weights.pkl", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# NOTE! NOTE! chose the loss function to use:
dist_mat = np.array([
[0., 1., 1., 1.],
[1., 0., 0.5, 0.5],
[1., 0.5, 0.,0.5],
[1., 0.5, 0.5, 0.]
])
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
#loss_fn = FocalLoss(weight=class_weights)
#loss_fn = MixedFocalLoss(alpha=class_weights)
#loss_fn = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

epoch_losses_train = []
epoch_losses_val = []
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):
        #current_time = time.time()

        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

        #print (time.time() - current_time)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
