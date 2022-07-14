
from torch.utils.data import Dataset
import json
import torch




def get_infos(self,item,ks):

    list_ = item['list']
    span = list(range (list_[0],list_[len(list_)-1]))
    #print(span)
    infos = self.data[str(0)] 
    data = infos['data']
    if ks is None:
        ks = data.keys()
    else:
        pass

    t = torch.zeros(len(span), len(ks))



    for relative_frame_position,i in enumerate(span):


        info = self.data[str(i)]

        data = info['data']
        
 
        
        #print('keys',ks)
        for j,key in enumerate(ks):
            #print('key',key)

            if key != "orimode":
                f = float(data[key][0])
                t[relative_frame_position,j] = f
            
        
    #print(t.shape)
    

    return t.float()



class OdometryDataset(Dataset):

    def __init__(self, dataset_path, keys = None):

        f = open(dataset_path)
        self.data = json.load(f)
        self.keys = keys
        

    def __len__(self):

        return len(self.data)

 
    def __getitem__(self, idx):

        item = self.data[str(idx)]
        if len(item['list']) == 1:
            
            return torch.zeros(10, 2),False
        else:
            
            data = get_infos(self, item, self.keys)

            return data, True
    