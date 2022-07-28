from __future__ import division
from errno import ESTALE
from itertools import count
import math
from select import select
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler
from tqdm import trange


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.samples_per_gpu = 1
        self.num_samples = int(
              math.ceil(len(self.dataset) * 1.0 / self.samples_per_gpu /
                        self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



class CustomSampler(Sampler):

    

    def __init__(self, data_source,sequence):

        print("FORECASTING SAMPLER")
    
        self.data_source = data_source
        self.sequence = sequence
        indices = list(range(len(self.data_source)))


        print('GENERATING INDICES')
        indices = self.convertbis(list(range(len(self.data_source))),self.sequence)
        print('FILTERING INDICES FROM DIFFERENT VIDEO-CLIPS')

        self.indices = self.filter_clips_efficient_all(indices,self.sequence)

    def __iter__(self):

       
        #gives in output list of list after filter process
        #we need to reacreate a unique list
        #filtered_indices = []
        
        #for i in self.indices:
            
        #    filtered_indices.extend(i)

        
        
        #only with preallocation
        filtered_indices = self.indices
        
        #print('INDICES------------',filtered_indices)
        #print('len', len(filtered_indices)/6)


        return iter(filtered_indices)



    def __len__(self):
        return len(self.data_source)


    def convertbis(self,indices,sequence):
        temp = []
        prior = []
        for i in trange(len(indices)):
            prior = []

            for ind in sequence:
                s = i+ind

                if s in range (0,len(indices)):
                    prior.append(s)

            if len(prior) == len(sequence):
                #print('index processing',i,'list added',prior)
                temp.extend(prior)
            else:
                #print('sequence in clip to short')
                pass

        return temp


    def filter_clips(self,temp,sequence):
        list_ = []
        pre_filter =[]
        final = []
        print('Creating list....')
        for i in trange(len(temp)):
            element = temp[i]       
            list_.append(element)
            if(len(list_)) == len(sequence):
                pre_filter.append(list_)
                list_ = []

        print('Filtering....')
        for i in trange(len(pre_filter)):
            list = pre_filter[i]
            mixed = False
            clip = self.data_source[list[0]]['clip']
            #print('processing',list,'clip',clip)

            for element in list:
                #print(self.data_source[element]['clip'])

                if not self.data_source[element]['clip']==clip :
                    #print('dont add this list')
                    mixed = True
                    break

            if mixed == True :
                #print('MIXED')
                continue
            else:
                final.append(list)

        return final


    def filter_clips_efficient(self,temp,sequence):
        
        list_ = []
        pre_filter =[]
        final = []
        print('Creating list....')
        for i in trange(len(temp)):
            element = temp[i]       
            list_.append(element)
            if(len(list_)) == len(sequence):
                pre_filter.append(list_)
                list_ = []

        print('Filtering....')
        for i in trange(len(pre_filter)):
            list = pre_filter[i]
            mixed = False
            clip = self.data_source[list[0]]['clip']
            #print('processing',list,'clip',clip)
            

            init = list[0]
            end = list[len(list)-1]

            clip_init = self.data_source[init]['clip']
            clip_end = self.data_source[end]['clip']
            

            if clip_init == clip_end :
                final.append(list)
            else:
                continue
        
        with open('/home/fperacch/Forecaster/temp/list.npy', 'wb') as f:
            final_ = np.array(final)

            np.save(f, final_)
        return final



    def filter_clips_efficient_all(self,temp,sequence):
        
        list_ = []
        pre_filter =[]
        print('Creating list....')
        for i in trange(len(temp)):
            element = temp[i]       
            list_.append(element)
            if(len(list_)) == len(sequence):
                pre_filter.append(list_)
                list_ = []

        print('Filtering....')

        final = [None]*(len(pre_filter)*6)
        last_id = 0


        for i in trange(len(pre_filter)):
            list = pre_filter[i]
            mixed = False
            clip = self.data_source[list[0]]['clip']
            #print('processing',list,'clip',clip)
            

            init = list[0]
            end = list[len(list)-1]

            clip_init = self.data_source[init]['clip']
            clip_end = self.data_source[end]['clip']
            

            if clip_init == clip_end :
                for i in range(len(list)):
                    final[last_id] = list[i]
                    last_id +=1
            else:
                continue
        

        final_fil = [i for i in final if i]
        final_fil.insert(0, 0)
        
        with open('/home/fperacch/Forecaster/temp/list.npy', 'wb') as f:
            final_ = np.array(final_fil)

            np.save(f, final_)
        return final_fil




class Kitti_Sampler(Sampler):

    def __init__(self, data_source):
        print("FORECASTING SAMPLER")
        self.data_source = data_source 
        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        return iter(self.indices)



    def __len__(self):
        return len(self.data_source)





class Kitti_DistSampler(Sampler):

    
    def __init__(self, data_source,rank,world_size):
        print("FORECASTING SAMPLER")
        self.data_source = data_source 
        indices = list(range(len(self.data_source)))
        self.indices = indices[rank:len(indices):world_size]

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.data_source)






   