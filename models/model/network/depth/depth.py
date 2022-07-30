

from turtle import forward
from .base import BaseDepth
import torch.nn as nn
from ...Depth.SGDepth.models.sgdepth import SGDepthDepth,SGDepthPose


##PS is pre-trained
#Use PS mask to improve training Depth estimation
#Use Depth forecasting hidden state to improve PS forecasting

#in summary :
#Depth estimation improved by PS
#PS forecasting improved by Depth forecasting


class DepthEsimator(nn.Module):

    def __init__(self, weights_init='pretrained', resolutions_depth=1, num_layers_pose=18,device_depth):
        super(DepthEsimator,self).__init__()
        self.device_depth = device_depth

        # While Depth and Seg Network have a shared Encoder,
        # each one has it's own Decoder
        self.depth = SGDepthDepth(self.common, resolutions_depth)

        # The Pose network has it's own Encoder ("Feature Extractor") and Decoder
        self.pose = SGDepthPose(
            num_layers_pose,
            weights_init == 'pretrained'
        )
        

    def forward():
        pass