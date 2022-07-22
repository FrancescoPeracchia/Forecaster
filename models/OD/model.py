import torch
import torch.nn as nn



class LSTM(nn.Module):

    def __init__(self, features,hidden_dim,layers):
        super(LSTM, self).__init__()

        self.features = features

        self.lstm = nn.LSTM(features, hidden_dim,num_layers=layers, batch_first = True)
        self.linear0 = nn.Linear(hidden_dim, 1)

    def forward(self, batch):


        input = batch[:,:4,:]
        gt = batch[:,4:,:]
        #print('gt',gt.shape)
        last_seen = input[:,3,:].detach().clone()
        #last_seen = gt[:,2,:].detach().clone()
        #print('input',input.shape)

        output,_ = self.lstm(input)
        #print('output after lstm',output.shape)
        output = output[:,-1,:]
        #print('output extract last ',output.shape)
        output = self.linear0(output)
        #print('output after linear',output.shape)


        #print('gt',gt.shape)
        #gt = torch.squeeze(gt, features)      
        #print('gt',gt.shape)
        gt = gt[:,-1,:]
        #print('gt',gt.shape)
        #print('output',output.shape)
        #print(gt)
        #print(output)

        
        
        #print('gt value',gt)
        #print('output value',output)
        #print('output shape',output.shape)


        return output,gt,last_seen