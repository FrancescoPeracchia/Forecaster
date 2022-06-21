import torch


def stack(features,depth,lengt):
    list = []
    list_= []
    for i in range(depth):
        #list of tensor to stack
        
        #len(features)-lengt beacuse only feature from 0 to 2 are from past images from 3 to 5 are futures

        for l in range(len(features)-lengt):
            list.append(features[l][0,i,:,:])
        

    for l in range(lengt):
        list_.append(features[int(l+lengt)][0,:,:,:])

    

    
    print('lenght of list shoulf be 256*3 and is',len(list))
    tensors = tuple(list)
    output = torch.stack(tensors,dim=0)
    print('output shape',output.shape)
    tensors = tuple(list_)
    output1 = torch.stack(tensors,dim=0)
    print('output shape',output1.shape)
    return output,output1

        

        