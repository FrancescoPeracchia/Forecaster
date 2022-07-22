from matplotlib.pyplot import cla
import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)


    def forward(self, features, previous_output, previous_hidden_state):

        input_gate = torch.sigmoid(self.Wxi(features) + self.Whi(previous_output))
        forget_gate = torch.sigmoid(self.Wxf(features) + self.Whf(previous_output))
        output_gate = torch.sigmoid(self.Wxo(features) + self.Who(previous_output))
        d = torch.tanh(self.Wxc(features) + self.Whc(previous_output))

        next_previous_hidden_state = forget_gate * previous_hidden_state + input_gate * d
        next_output = output_gate * torch.tanh(next_previous_hidden_state)
        return next_output, next_previous_hidden_state

class Level_ConvLSTM(nn.Module):
    def __init__(self,channel_dim,kernel_size):
        super(Level_ConvLSTM).__init__()

    

    def forward(self,input):
        """
        Args: Input tensor of shape (B,C,H,W) i.e : (3,256,128,64), H,W are changing accordigly with the level resolution

        Return: Input tensor of shape (B,C,H,W) i.e : (1,256,128,64)

        """

        return


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            print(i)

    def forward(self, input):
        internal_state = []
        outputs = []
        #for every step we have a new real features input
        for step in range(self.step):
            x = input
            
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

from torchsummary import summary

if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512, hidden_channels=[124,64], kernel_size=3, step=5,
                        effective_step=[4]).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 512, 64, 32)).cuda()
    target = Variable(torch.randn(1, 128, 64, 32)).double().cuda()

   
    output = convlstm(input)
    output = output[0][0].double()
    #res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-10, raise_exception=True)
    #print(res)
    summary(convlstm,( 512, 64, 32))

   