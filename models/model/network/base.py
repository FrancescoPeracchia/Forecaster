from abc import ABCMeta, abstractmethod
from typing import List

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import torch
from mmdet.core import auto_fp16, get_classes, tensor2imgs
from mmdet.utils import print_log


class BaseForecaster(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self,defice_ps,device_depth):
        super(BaseForecaster, self).__init__()
        
        self.device_ps = defice_depth
        self.device_depth = device_depth
        
    



    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
    
    
    @abstractmethod
    def forward_train(self, list_image,list_id, targets, **kwargs):
        pass
    
    @abstractmethod
    def forward_test(self, list_image,list_id, targets, **kwargs):
        pass

       
    @auto_fp16(apply_to=('img', ))
    def forward(self, input, targets, return_loss=True, eval=None, **kwargs):
        

        
        ids = input['ids']
        imgs = input['img']
                
       
        list_image = []
       
        
        for image in imgs:
               
            #print('shape',image.shape)
            image = torch.squeeze(image, 0)
            image = torch.transpose(image, 0, 2)   
            #print('shape',image.shape)
            image = torch.transpose(image, 1, 2)
            #print('shape',image.shape)
            im = torch.unsqueeze(image, 0).float().to(self.device)
            #print(im.shape)
            list_image.append(im)

   

        if return_loss:
            return self.forward_train(list_image,ids, targets, **kwargs)
        else:
            return self.forward_test(list_image,ids, targets, **kwargs)

