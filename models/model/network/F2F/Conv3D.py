from requests import PreparedRequest
from .base import BasePredictor
import torch.nn as nn
import torch

class Conv3D(BasePredictor):
    def __init__(self,forecaster_cfg):
        super(BasePredictor, self).__init__()
        self.list_key = {'0':'low','1':'medium','2':'high','3':'huge'}
        self.inv_list_key = {'low':'0','medium':'1','high':'2','huge':'3'}

        
        self.kernel_low = forecaster_cfg.low.kernel
        self.stride_low = forecaster_cfg.low.stride
        self.padding_low = forecaster_cfg.low.padding
        self.skip_low = forecaster_cfg.low.skip

        self.kernel_medium = forecaster_cfg.medium.kernel
        self.stride_medium = forecaster_cfg.medium.stride
        self.padding_medium = forecaster_cfg.medium.padding
        self.skip_medium = forecaster_cfg.medium.skip


        self.kernel_high = forecaster_cfg.high.kernel
        self.stride_high = forecaster_cfg.high.stride
        self.padding_high = forecaster_cfg.high.padding
        self.skip_high = forecaster_cfg.high.skip

        self.kernel_huge = forecaster_cfg.huge.kernel
        self.stride_huge = forecaster_cfg.huge.stride
        self.padding_huge = forecaster_cfg.huge.padding
        self.skip_huge = forecaster_cfg.huge.skip


        
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        list_devices = [cuda0,cuda1,cuda1,cuda1]

        
        self.f2f_low = F2F_base(256,256,self.kernel_low,self.stride_low,self.padding_low,self.skip_low,list_devices[0]).to(list_devices[0])
        self.f2f_medium = F2F_base(256,256,self.kernel_medium,self.stride_medium,self.padding_medium,self.skip_medium,list_devices[1]).to(list_devices[1])
        self.f2f_high = F2F_base(256,256,self.kernel_high,self.stride_high,self.padding_high,self.skip_high,list_devices[2]).to(list_devices[2])
        self.f2f_huge =F2F_base(256,256,self.kernel_huge,self.stride_huge,self.padding_huge,self.skip_huge,list_devices[3]).to(list_devices[3])

        
        
        


        self.loss = nn.L1Loss()
        



    def forward_train(self,past,future,targets):

        """
        Args: Dictionaty with {'0':'low','1':'medium','2':'high','3':'huge'}
        Dictionaty['low'] is a torch tensor  of size ([D*N,256,512]) i.e ([768,256,512])

        Return: Loss for each level
        """
        #print(past['low'].shape)
        #print(past['medium'].shape)
        #print(past['high'].shape)
        #print(past['huge'].shape)
        
        l_shape = past['low'].shape
        m_shape = past['medium'].shape
        b_shape = past['high'].shape
        h_shape = past['huge'].shape
        losses = {}


        low = torch.reshape(past['low'],(256,3,l_shape[1],l_shape[2]))
        #print(low[0,0,:,:])
        #print(past['low'].shape)
        medium = torch.reshape(past['medium'],(256,3,m_shape[1],m_shape[2]))
        #print(past['low'].shape)
        high = torch.reshape(past['high'],(256,3,b_shape[1],b_shape[2]))
        #print(past['low'].shape)
        huge = torch.reshape(past['huge'],(256,3,h_shape[1],h_shape[2]))

        #Network
        low_pre = self.f2f_low(low)
        medium_pre = self.f2f_medium(medium)
        high_pre = self.f2f_high(high)
        huge_pre = self.f2f_huge(huge)
        

       #from torch([1,256,1,512,256])
       # to  torch([1,1,256,512,256])

        low_pre = torch.transpose(low_pre, 1, 2)
        medium_pre = torch.transpose(medium_pre, 1, 2)
        high_pre = torch.transpose(high_pre, 1, 2)
        huge_pre = torch.transpose(huge_pre, 1, 2)
        

        predictions = {'low':low_pre,'medium':medium_pre,'high':high_pre,'huge':huge_pre}


        for feature in range(len(future)) :
            
            key = self.list_key[str(feature)]
            #key could be 'low' 'medium' 'high' 'huge'
             
            pre = predictions[key]
            pre = torch.squeeze(pre, 0)
            #from torch([1,1,256,W,H])
            #to torch([1,256,W,H])


            #considering only the first image for training
            fu = future[key][0,:,:,:]
            fu = torch.unsqueeze(fu, 0)
            #from torch([3,256,W,H])
            #to torch([1,256,W,H])
            #and then torch([1,256,W,H])

            #bothe predicted and future have the same shape
            #to compute the loss we have to check that both the
            #tensors are in the same GPU




            if pre.get_device() != fu.get_device():
                fu = fu.to(pre.get_device())


            #print(fu.get_device())
            #print(pre.get_device())

            
            #to avoid mistake in DEBUGGING
            assert pre.shape == fu.shape, 'size between prediceted features and future features are not matching '
            assert pre.get_device() == fu.get_device(), 'prediceted features and future features are not in the same GPU'
            
            #Loss is finalli computed and stored under the relative key
            #each key ll have a different loss :'low' 'medium' 'high' 'huge'
            #then in train.py ll be computed a comulative loss
            loss_ = self.loss(pre, fu)
            losses[str(key)]  = loss_

       
        return losses

    def forward_test(self,past,future,targets):
        """
        Args: Dictionaty with {'0':'low','1':'medium','2':'high','3':'huge'}
        Dictionaty['low'] is a torch tensor  of size ([D*N,256,512]) i.e ([768,256,512])

        Return: List features for both predicted target frame and gt target frame
        """



        low = torch.reshape(past['low'],(256,3,256,512))
        #print(low[0,0,:,:])
        #print(past['low'].shape)
        medium = torch.reshape(past['medium'],(256,3,128,256))
        #print(past['low'].shape)
        high = torch.reshape(past['high'],(256,3,64,128))
        #print(past['low'].shape)
        huge = torch.reshape(past['huge'],(256,3,32,64))

        #Network
        low_pre = self.f2f_low(low)
        medium_pre = self.f2f_medium(medium)
        high_pre = self.f2f_high(high)
        huge_pre = self.f2f_huge(huge)
        

       #from torch([1,256,1,512,256])
       # to  torch([1,1,256,512,256])

        low_pre = torch.transpose(low_pre, 1, 2)
        medium_pre = torch.transpose(medium_pre, 1, 2)
        high_pre = torch.transpose(high_pre, 1, 2)
        huge_pre = torch.transpose(huge_pre, 1, 2)
        

        predictions = {'low':low_pre,'medium':medium_pre,'high':high_pre,'huge':huge_pre}
        pre_out = {}
        gt_out = {}


        for feature in range(len(future)) :
            
            key = self.list_key[str(feature)]
            #key could be 'low' 'medium' 'high' 'huge'
             
            pre = predictions[key]
            pre = torch.squeeze(pre, 0)
            #from torch([1,1,256,W,H])
            #to torch([1,256,W,H])


            #considering only the first image for training
            fu = future[key][0,:,:,:]
            fu = torch.unsqueeze(fu, 0)
            #from torch([3,256,W,H])
            #to torch([1,256,W,H])
            #and then torch([1,256,W,H])

            if pre.get_device() != fu.get_device():
                fu = fu.to(pre.get_device())

  
            #to avoid mistake in DEBUGGING
            assert pre.shape == fu.shape, 'size between prediceted features and future features are not matching '
            assert pre.get_device() == fu.get_device(), 'prediceted features and future features are not in the same GPU'

            pre_out[key] = pre
            gt_out[key] = fu
            


       
        return pre_out,gt_out





class F2F_base(nn.Module):

    def __init__(self,input_channel,output_channel,kernel,stride,padding,skip = False,device = None):
        super(F2F_base,self).__init__()

        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel, stride = stride, padding = padding)
        self.skip = skip
        self.device = device
        
        

    def forward(self,x):
        x = torch.unsqueeze(x, 0)
        x = x .to(self.device)
        #print('shape',x.shape)


        y = self.conv1(x)
        if self.skip == False: 
            y =torch.add(y, x[0,:,0,:,:])
  
        return y

  