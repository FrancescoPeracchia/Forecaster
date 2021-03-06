from abc import ABCMeta, abstractmethod
import pstats

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import torch
from mmdet.utils import print_log
from .utils import stack


class BasePredictor(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self,forecaster_cfg):
        super(BasePredictor, self).__init__()
        self.device = 'cuda:1'
        self.list_key = {'0':'low','1':'medium','2':'high','3':'huge'}

    



    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
    
    
    @abstractmethod
    def forward_train(self):

        pass
    
    @abstractmethod
    def forward_test(self):

        pass


    
    def forward(self, input, targets, return_loss=True):
        
        past,future = self.concatenate_by_depth(input,targets)
        #print(type(past['low']))
        #print(type(future['low']))
        

        

        if return_loss:
            return self.forward_train(past,future,targets)
        else:
            return self.forward_test(past,future,targets)
    

    def concatenate_by_depth(self,input,target):
        """
        Args :input is a Dictionary where each element is list of features for each images
        [D,W,H]....each image


        Return :a list where each element had depth concatenated features
        N. number of past images, target is used to don't consider features from futures images
        [N*D,W H]
        
        
        """
        lengt = len(target)
        #print(len(input['low']))
        #i.e = 3 sono 3+3 = 6
        input_features_INPUT = {}
        input_features_FUTURE = {}

        for feature_level in range(len(input)):
            

            #4
            features = input[str(self.list_key[str(feature_level)])]
            
            #read from image 1 the depth dimetions [N,D,H,W]
            depth = features[0].shape[1]
            #print('depth',depth)
            
            feature,future = stack(features,depth,lengt)
            #each feature has shape
            #torch.Size([768, 256, 512]),
            #torch.Size([768, 128, 256]),
            #torch.Size([768, 64, 128]),
            #torch.Size([768, 32, 64])
            input_features_INPUT[str(self.list_key[str(feature_level)])] = feature
            input_features_FUTURE[str(self.list_key[str(feature_level)])] = future 



    
        return input_features_INPUT,input_features_FUTURE
    

    




