from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import torch
from mmdet.core import auto_fp16, get_classes, tensor2imgs
from mmdet.utils import print_log


class BaseForecaster(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseForecaster, self).__init__()
        self.device = 'cuda:0'
    



    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
    
    
    @abstractmethod
    def forward_train(self, list_image,list_id, targets, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass
    
    @abstractmethod
    def forward_test(self, list_image,list_id, targets, **kwargs):
        pass

       
    @auto_fp16(apply_to=('img', ))
    def forward(self, input, targets, return_loss=True, eval=None, **kwargs):

        #from torch.Size([3, 1024, 2048, 3])
        #to list of  torch.Size([1024, 2048, 3])
        imgs = input['img']        
        ids = input['id']
        list_image = []
        list_id = []        
        for image in range(imgs.size()[0]):
            
            im = imgs[image,:,:,:]           
            #print('shape',im.shape)
            im = torch.transpose(im, 0, 2)
            im = torch.transpose(im, 1, 2)
            im = torch.unsqueeze(im, 0).float().to(self.device)
            #print('shape',im.shape)
            list_image.append(im)
            list_id.append(int(ids[image]))
        


        if return_loss:
            return self.forward_train(list_image,list_id, targets, **kwargs)
        else:
            return self.forward_test(list_image,list_id, targets, **kwargs)

