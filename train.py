from __future__ import division
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from mmcv import Config
from PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from models.model import build_forecaster
from PS.mmdet.apis import set_random_seed
from datetime import datetime
from utils.utils_train import train_one_epoch, validation, get_lr
import numpy as np
import os
from torchsummary import summary


from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR




def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('output', help='output path for saving the model')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
   

    args = parser.parse_args()


    return args


def main():
    directory = os.getcwd()

    print('Current directory : ',directory)
    args = parse_args()

    output_path = args.output


    cfg = Config.fromfile(args.config)

    #Update the relative path 
    cfg.model['efficientPS_config'] = str(directory+cfg.model['efficientPS_config'])
    print('Config path :', cfg.model['efficientPS_config'])

    cfg.model['efficientPS_checkpoint'] = str(directory+cfg.model['efficientPS_checkpoint'])
    print('PS Checkpoint path loaded :', cfg.model['efficientPS_checkpoint'])

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    cfg.seed = args.seed
    

    model = build_forecaster(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    

    summary(model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/MODEL_1_TRAINING{}'.format(timestamp))

    
    
    #Update the relative path 
    cfg.data.train['ann_file'] = str(directory+cfg.data.train['ann_file'])
    print('ann_file path :', cfg.data.train['ann_file'])

    cfg.data.train['data_root'] = str(directory+cfg.data.train['data_root'])
    print('data_root path :', cfg.data.train['data_root'])

    cfg.data.validation['ann_file'] = str(directory+cfg.data.validation['ann_file'])
    print('ann_file path :', cfg.data.validation['ann_file'])

    cfg.data.validation['data_root'] = str(directory+cfg.data.validation['data_root'])
    print('data_root path :', cfg.data.validation['data_root'])
    

    output_path = os.path.join(output_path,'model_predictor.pth')

    #lists of datesets
    datasets = [build_dataset(cfg.data.train)]
    datasets_val = [build_dataset(cfg.data.validation)]



    #lists of dataloaders
    data_loaders = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets
    ]

    
    data_loaders_val = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets_val
    ]

    #batch = next(iter(data_loaders))
    #print(batch)

    #for i, data in enumerate(data_loaders[0]):
    #    print(data)





    optimizer = torch.optim.SGD(model.predictor.parameters(), lr=0.5, momentum=0.9)
    scheduler0 = ExponentialLR(optimizer, gamma=0.9)
    scheduler1 = MultiStepLR(optimizer, milestones=[10,25,45], gamma=3)



    best_avg_loss_val = 9999
    epoch_number = 0
    EPOCHS = 70
    skip = True

    for epoch in range(EPOCHS):

        if skip :
            print('EPOCH {}:'.format(epoch_number + 1))

            #YOU SHOULDN'T CALL model.train() model contains both efficientPS and Forecaster 
            #it ll generatate a model with efficientPS parts for training activated
            model.predictor.train()
            avg_loss,loss_dict = train_one_epoch(cfg, model, data_loaders, optimizer)
            print('avg_loss',avg_loss)


    

            #TENSORBOARD GRAPH TRAIN
            for log in loss_dict:
                mean = np.mean(loss_dict[log])
                t= os.path.join('Loss',str(log))
                writer.add_scalar(t,mean,epoch)
        




        #SCHEDULER & TENSORBOARD GRAPH LR
        lr = get_lr(optimizer)
        scheduler0.step()
        scheduler1.step()
        writer.add_scalar('learning rate',lr,epoch)
        
        

        model.eval()
        #note model.efficientps is already at eval() you could even use model.predictor.eval()
        with torch.no_grad():
            print('VALIDATION ')
            avg_loss_val,loss_dict_val = validation(cfg,model,data_loaders_val)
            print('avg_loss_val',avg_loss_val)

            #TENSORBOARD GRAPH VAL
            for log in loss_dict_val:
                mean = np.mean(loss_dict_val[log])
                t= os.path.join('Loss_validation',str(log))
                writer.add_scalar(t,mean,epoch)
        
                


        epoch_number += 1

        if avg_loss_val < best_avg_loss_val:
            best_avg_loss_val = avg_loss_val
            torch.save(model.predictor.state_dict(),output_path)
    


if __name__ == '__main__':
    main()

