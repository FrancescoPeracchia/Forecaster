from __future__ import division
import argparse
import copy
import warnings
from xmlrpc.client import boolean
from models.PS.mmdet.models import efficientps
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from mmcv import Config
from models.PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from models.model import build_forecaster
from models.PS.mmdet.apis import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.utils_train import validation
from utils.save_predictions import save_prediction
from utils.panoptic_evaluation import evaluate
import os
import numpy as np
from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('modality', default='Validation', help='Test or Save, testing gives you in outuput metrics about Test dataset, Save save the results in the specified folder')
    parser.add_argument('--output', help='output path')
    parser.add_argument('--gt',type=bool, help= 'Used with Save : modality, True if you want to produce in output also the gt map\
                        ,False if is already computed and you want to generate only results from predictions')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()
    return args


def main():

    directory = os.getcwd()

    print(directory)
    args = parse_args()

    #config and from args to config.test
    cfg = Config.fromfile(args.config)
    
    if args.output is not None:    
        cfg.test['output_path'] = args.output

    if args.gt is not None:      
        cfg.test['GT'] = args.gt


    Modality = args.modality

    #Update the relative path 
    cfg.model['efficientPS_config'] = str(directory+cfg.model['efficientPS_config'])
    print('Config path :', cfg.model['efficientPS_config'])

    cfg.model['efficientPS_checkpoint'] = str(directory+cfg.model['efficientPS_checkpoint'])
    print('Checkpoint path :', cfg.model['efficientPS_checkpoint'])

    #build forecaster with pretrained predictor
    model = build_forecaster(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, eval = True)
    
    #it should be already in eval mode
    model.eval()


    summary(model)
    
    
    #Update the relative paths 
    cfg.data.validation['ann_file'] = str(directory+cfg.data.validation['ann_file'])
    print('ann_file path :', cfg.data.validation['ann_file'])

    cfg.data.validation['data_root'] = str(directory+cfg.data.validation['data_root'])
    print('data_root path :', cfg.data.validation['data_root'])

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    cfg.seed = args.seed

   

    #lists of datesets
   
    datasets_val = [build_dataset(cfg.data.validation)]

    data_loaders_val = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets_val]



    
    
    #note model.efficientps is already at eval() you could even use model.predictor.eval()
    if Modality == 'Validation' :
    
        with torch.no_grad():
            print('VALIDATION ')

            panoptics_result = evaluate(cfg, model, data_loaders_val,datasets_val)


            #avg_loss_val,loss_dict_val = validation(cfg, model,data_loaders_val)
            #print('avg_loss_val',avg_loss_val)

            #for log in loss_dict_val:
                #mean = np.mean(loss_dict_val[log])
                #print('mean loss',log,' is : ',mean)
            
            
                

    elif Modality == 'Save':

        save_prediction(cfg, model, data_loaders_val, datasets_val)

    else :
    
            print("No valid modality used, only Save and Test avaible") 
            return
    

if __name__ == '__main__':
    main()
