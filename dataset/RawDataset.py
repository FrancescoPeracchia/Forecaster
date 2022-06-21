
from operator import index
from time import process_time_ns
from torch.utils.data import Dataset
import json
import numpy as np
from PS.mmdet.datasets.pipelines import Compose
import mmcv

class LoadImage(object):

        def __call__(self, results):
            if isinstance(results['img_prefix'], str):
                results['filename_complete'] = results['img_prefix']+'/'+results['img_info']['filename']
            else:
                results['filename'] = None
            img = mmcv.imread(results['filename_complete'])
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            return results


class RawDataset(Dataset):

    def __init__(self,
                ann_file,
                pipeline,
                data_root=None,
                test_mode=False):

        self.ann_file = ann_file
        self.data_root = data_root
        self.pipeline = pipeline
        self.test_mode = test_mode
        


        self.img_infos = self.load_annotations(self.ann_file)
        #print('Annotation',self.img_infos)
        
        if not self.test_mode:
            self._set_group_flag()



        
        

    def __len__(self):
        return len(self.img_infos)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            self.flag[i] = 1

    def load_annotations(self, ann_file):
        
            # Opening JSON file
            #print(ann_file)
            f = open(ann_file)         
            return json.load(f)


    def get_ann_info(self, idx):
        return self.img_infos[idx]


    def __getitem__(self, idx):
            data=dict()
            
        
            #print(self.img_infos[idx])
            data['img'] = self.prepare_train_img(idx)['img']
            data['clip'] = self.img_infos[idx]['img_info']['end_frame']
            data['id'] = self.img_infos[idx]['img_info']['id']
                
                   
            return data
    
  


    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        #print('indeces',img_info)
        test_pipeline = [LoadImage()]+ self.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(img_info)
        return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

 