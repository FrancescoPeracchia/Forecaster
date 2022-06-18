from __future__ import division
import argparse
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mmcv import Config
from PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from model import build_forecaster
from PS.mmdet.apis import set_random_seed



import os

#to get the current working directory



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()


    return args


def main():
    directory = os.getcwd()

    print(directory)
    args = parse_args()

    cfg = Config.fromfile(args.config)

    #Update the relative path 
    cfg.model['efficientPS_config'] = str(directory+cfg.model['efficientPS_config'])
    print('Config path :', cfg.model['efficientPS_config'])

    cfg.model['efficientPS_checkpoint'] = str(directory+cfg.model['efficientPS_checkpoint'])
    print('Checkpoint path :', cfg.model['efficientPS_checkpoint'])

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    cfg.seed = args.seed
    

    model = build_forecaster(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    
    #Update the relative path 
    cfg.data.train['ann_file'] = str(directory+cfg.data.train['ann_file'])
    print('ann_file path :', cfg.data.train['ann_file'])

    cfg.data.train['data_root'] = str(directory+cfg.data.train['data_root'])
    print('data_root path :', cfg.data.train['data_root'])

    datasets = [build_dataset(cfg.data.train)]

    


    data_loaders = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets
    ]


    
    iter_data =  iter(data_loaders[0])
    print('fine iter')
    
    train_image = next(iter_data)
    print('train_image',train_image)

    print(len(model(train_image,cfg.modality['target'])['huge']))
    print(model(train_image)['medium'][0].size())
    print(cfg.modality['target'])


    '''
    device = next(model.parameters()).device

    
    print(train_image.size())
    
    x = train_image[0]
    save_image(x)
    
    
    #model(train_image)

    '''
 




if __name__ == '__main__':
    main()
