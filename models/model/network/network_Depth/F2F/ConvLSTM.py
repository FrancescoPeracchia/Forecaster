from ast import literal_eval
from turtle import forward, position

from zmq import device
import torch
import torch.nn as nn
from torch.autograd import Variable
from .base import BasePredictor

def read_previous_inter_hidden_states(n,list):
    list_previous = []
    #n is between 1 and 4, for 1 : should be loaded only 0, for 2 : 0 and 1....
    for i in range(n):
        list_previous.append(list[i])
    
    return list_previous



class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        if self.Wci is None:
            #print( hidden, shape[0], shape[1])
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device))

class PathWay(nn.Module):

    def __init__(self, level, input_channels, hidden_channels, kernel_size, device):
        super(PathWay, self).__init__()

        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.conv_list = []
        self.upsampler_list = []
        self.level = level
        self.scale = 2**level
        self.device = device 

        for i in range(level):
            name = 'PathConv{}{}'.format(i,self.level)
            cell = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True).to(self.device)
            setattr(self, name, cell)
            self.conv_list.append(cell)
            name_upsampler = 'Upsampler{}{}'.format(i,level)
            upsampler = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=True)
            self.scale =(self.scale/2)
            
            setattr(self, name_upsampler, upsampler)
            self.upsampler_list.append(upsampler)
            
    def forward(self,previous_hidden,inter_level_hiddens):
    
        
        bsize, _, height, width = previous_hidden.size()
        output = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width)).to(self.device)
       

        
        for i,inter_level in enumerate(inter_level_hiddens):
            name = 'Upsampler{}{}'.format(i,self.level)
            up = getattr(self, name)
            #print('Upsampler : ',up)
            inter_level_upsampled = getattr(self, name)(inter_level)
            name = 'PathConv{}{}'.format(i,self.level)
            inter_level = getattr(self, name)(inter_level_upsampled)
            Avl = torch.sigmoid(inter_level)
            Avl = torch.mul(Avl, inter_level_upsampled)
            output= torch.add(output, Avl)


        c = torch.add(previous_hidden, output)


        return c

class ConvLSTM(nn.Module):
 
    def __init__(self, input_channels, hidden_channels, kernel_size, level, device):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self._all_layers = []
        self.level = level
        self.device = device

        if self.level > 0 :
            self.name_pathway = 'pathway'
            pathway = PathWay(self.level, self.input_channels[0], self.hidden_channels[0], self.kernel_size, self.device).to(self.device)
            setattr(self, self.name_pathway, pathway)


        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.device).to(self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
          
    def forward(self, input, step, layer = None, inter = False , inter_values = None):

        x = input.to(self.device)
        

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            #print('cell forward :',name,'layer n.',i,' step n.',step)
            
            # all cells are initialized in the first step
            if step == 0 :
                self.internal_state = []
                self.outputs = []
                bsize, _, height, width = x.size()
                self.hidden(name,bsize,height, width,i)

            
            #read previous internal state
            (h, c) = self.internal_state[i]

            if inter and layer>0 and step > 0:
                c = getattr(self,self.name_pathway)(c,inter_values)

            x, new_c = getattr(self, name)(x, h, c)
            self.internal_state[i] = (x, new_c)
  
        return (x, new_c)
    
    def hidden(self,name,bsize,height,width,i):
        
            
            (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                        shape=(height, width))
            self.internal_state.append((h, c))

class MultiConvLSTM(BasePredictor):

    def __init__(self,forecaster_cfg, device = None):
        super(BasePredictor, self).__init__()
        print(forecaster_cfg)

        self.list_key = {'0':'huge','1':'high','2':'medium','3':'low'}
        self.inv_list_key = {'huge':'0','high':'1','medium':'2','low':'3'}
       

        self.channel_siz_input = forecaster_cfg.channel_size
        self.channel_siz_output = [self.channel_siz_input]
        self.inter_level = forecaster_cfg.inter_level
        self.kernels = forecaster_cfg.kernels
        self.num_layers = len(self.kernels)
        self.step = forecaster_cfg.step
        self.effective_step = forecaster_cfg.effective_step
        self._all_levels = []

        self.levels = [i for i in range(len(self.kernels))]
        self.loss = nn.L1Loss()
        self.device = device



        self.cuda0 = torch.device('cuda:0')
        self.cuda1 = torch.device('cuda:1')

        if self.device != None :
            self.list_devices = [self.device for i in range(self.num_layers)]
        else :
            self.list_devices = [self.cuda0,self.cuda0,self.cuda0,self.cuda0]
        print('Device List',self.list_devices)
        



        for i in range(self.num_layers):
            name = 'ConvLSTM_level{}'.format(i)

            level = ConvLSTM(input_channels = self.channel_siz_input, hidden_channels= self.channel_siz_output, kernel_size = self.kernels[i], level = self.levels[i], device = self.list_devices[i])
            setattr(self, name, level)
            self._all_levels.append(level)

    def forward_train(self,input,future,targets):
        
        
        """
        Args: Dictionaty with {'0':'low','1':'medium','2':'high','3':'huge'}
        Dictionaty['low'] is a torch tensor  of size ([D*N,256,512]) i.e ([768,256,512])

        Return: Loss for each level
        """

        outputs = []
        #init intra-level connection
        new_inter_hidden_states = [0]* (len(input))
        inter_hidden_states = [0]* (len(input))
        losses = {}
        

        for step in range(self.step):
         
        
            for n,input_level in enumerate(input):

                #print('------------------------------------')
                name = 'ConvLSTM_level{}'.format(n)
                #print('name : ',name,' level n.',n, 'for step n.',step, ' input', input_level.size())
                
            
                x = input_level[:,step,:,:,:]            
                #layer 0 does't receive hidden state from previous layers
                model_level = getattr(self, name)
                #print('device : ',model_level.device)  
                if self.inter_level:
                    if n > 0 and step > 0:
                        #print('')
                        inter_level_H = read_previous_inter_hidden_states(n,inter_hidden_states)                          
                        (x, new_c) = model_level(x, step, layer = n, inter = self.inter_level, inter_values = inter_level_H)          
                        new_inter_hidden_states[n] = new_c
                    
                    else:
                        #print('STEP0 or CONV0')
                        (x, new_c) = model_level(x, step, layer = n, inter = self.inter_level)
                        new_inter_hidden_states[n] = new_c

                
                else :
                    (x, new_c) = model_level(x, step)

                for m,effective in enumerate(self.effective_step):

                    if step == effective:
                        outputs.append(x)
                        #print('output size saved : ',x.size())
                        current_level_future = future[n][:,0,:,:,: ].to(self.list_devices[0])
                        loss_ = self.loss(x, current_level_future)
                        level_string = 'level'+str(n)
                        losses[str(self.list_key[str(n)])]=loss_
                
            #print('NEW HIDDEN STATE ARE PAST')
            inter_hidden_states = new_inter_hidden_states


        return losses

    
    def forward_test(self,input,future):
        return

"""
    
if __name__ == '__main__':
    # gradient check
    convlstm = MultiConvLSTM(channel_size = 256,kernels= [3,3,3,3], step = 3, effective_step=[2], inter_level = True).cuda()
    input1 = Variable(torch.randn(1,3, 256, 16, 16)).cuda()
    input2 = Variable(torch.randn(1,3, 256, 32, 32)).cuda()
    input3 = Variable(torch.randn(1,3, 256, 64, 64)).cuda()
    input4 = Variable(torch.randn(1,3, 256, 128, 128)).cuda()
    input = [input1,input2,input3,input4]
    print(convlstm)

    output = convlstm(input)
    print('____________________________________________________')
    output = convlstm(input)
    print(len(output))
    print(output[3].size())
    pytorch_total_params = sum(p.numel() for p in convlstm.parameters() if p.requires_grad)
    print(pytorch_total_params)
    #print(convlstm)

"""



