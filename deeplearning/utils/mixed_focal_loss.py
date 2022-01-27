import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixedFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75):
      super(MixedFocalLoss, self).__init__()
      self.weight = weight #represents lambda parameter and controls weight given to Focal Tversky loss and Focal loss
      self.alpha = alpha #controls weight given to each class
      self.beta = beta #controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives.
      self.delta = delta
      self.gamma_f = gamma_f #modified Focal loss' focal parameter controls degree of down-weighting of easy examples
      self.gamma_fd = gamma_fd #modified Focal Dice loss' focal parameter controls degree of down-weighting of easy examples

    def focal_dice_loss(self,y_true, y_pred, delta=0.7, gamma_fd=0.75):
      smooth=0.000001 #smoothing constant to prevent division by 0 errors
      # Clip values to prevent division by zero error
      epsilon = 1e-7
      y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
      axis = [1,2]
      # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
      tp = torch.sum(y_true * y_pred, axis=axis)
      fn = torch.sum(y_true * (1-y_pred), axis=axis)
      fp = torch.sum((1-y_true) * y_pred, axis=axis)
      dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
      # Sum up classes to one score
      focal_dice_loss = torch.sum(torch.pow((1-dice_class), gamma_fd), axis=[-1])
  	# adjusts loss to account for number of classes
      num_classes = 4
      focal_dice_loss = focal_dice_loss / num_classes
      focal_dice_loss = focal_dice_loss.mean()
      return focal_dice_loss
    
    def focal_loss(self,target, input, alpha=None, beta=None, gamma_f=2.):
      ce_loss = F.cross_entropy(input, target,reduction='mean',weight=self.alpha)
      pt = torch.exp(-ce_loss)
      focal_loss = ((1 - pt) ** self.gamma_f * ce_loss).mean()
      return focal_loss
    
    def forward(self, input, target):
      # Obtain Focal Dice loss
      focal_dice = self.focal_dice_loss(target,input)
      # Obtain Focal loss
      focal = self.focal_loss(target,input,alpha=self.alpha)
      # return weighted sum of Focal loss and Focal Dice loss
      if self.weight is not None:
            return (self.weight * focal_dice) + ((1-self.weight) * focal)  
      else:
          return focal_dice + focal