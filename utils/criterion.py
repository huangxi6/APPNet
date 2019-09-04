import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d
#from .dataset.target_generation import generate_edge
import pdb

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
   
    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        #pdb.set_trace()
        loss = 0

        # loss for parsing 1
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])
        
        #loss for parsing 2
        preds_parsing = preds[1]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[1])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[1])

        return loss

    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        #pdb.set_trace()

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        # scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # loss3 = self.criterion2(scale_pred, target)
        return loss1 + loss2*0.4

    