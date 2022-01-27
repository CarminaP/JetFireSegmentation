import sys

sys.path.append("/root/deeplearning")
from datasets import DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("/root/deeplearning/model")
from deeplabv3 import DeepLabV3
from unet import UNet
from segnet import SegNet
from attentionUnet import AttU_Net
from unetpp import UnetPlusPlus

sys.path.append("/root/deeplearning/utils")
from utils import label_img_to_color

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

batch_size = 1
model_id = 1

# NOTE! NOTE! chose the architecture to train:
#network = DeepLabV3(model_id, project_dir="/root/deeplearning").cuda()
#network = UNet(model_id, project_dir="/root/deeplearning").cuda()
#network = SegNet(model_id, project_dir="/root/deeplearning").cuda()
#network = AttU_Net(model_id, project_dir="/root/deeplearning").cuda()
network = UnetPlusPlus(model_id, project_dir="/root/deeplearning").cuda()

network.load_state_dict(torch.load("/root/deeplearning/training_logs/model_"+str(model_id)+"/checkpoints/model_"+str(model_id)+"_epoch_1349.pth")) #Change this to the newly trained pth

val_dataset = DatasetEval(data_path="/root/deeplearning/data/jetfire")

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
for step, (imgs, img_ids) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            if i == 0:
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                img = imgs[i] # (shape: (3, img_h, img_w))

                img = img.data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img = img*np.array([0.22379429, 0.25679154, 0.19004982])
                img = img + np.array([0.06254871, 0.09143332, 0.53944639])
                img = img*255.0
                img = img.astype(np.uint8)

                pred_label_img_color = label_img_to_color(pred_label_img)
                #overlayed_img = 0.35*img + 0.65*pred_label_img_color
                #overlayed_img = overlayed_img.astype(np.uint8)
                pred_label_img_color = cv2.resize(pred_label_img_color, (640, 480),
                         interpolation=cv2.INTER_NEAREST)
                pred_label_img_color = pred_label_img_color.astype(np.uint8)

                cv2.imwrite(network.model_dir + "/segmentation/" + img_id + "_segmask.png", pred_label_img_color)
                #cv2.imwrite(network.model_dir + "/" + img_id + "_overlayed.png", overlayed_img)
