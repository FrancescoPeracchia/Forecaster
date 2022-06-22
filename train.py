from __future__ import division
import argparse
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from mmcv import Config
from PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from model import build_forecaster
from PS.mmdet.apis import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_train import parse_loss

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
    print(iter_data)

    #optimizer_low = torch.optim.SGD(model.predictor.f2f_low.parameters(), lr=0.001, momentum=0.9)
    #optimizer_medium = torch.optim.SGD(model.predictor.f2f_medium.parameters(), lr=0.001, momentum=0.9)
    #optimizer_high = torch.optim.SGD(model.predictor.f2f_high.parameters(), lr=0.001, momentum=0.9)
    #optimizer_huge = torch.optim.SGD(model.predictor.f2f_huge.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.SGD(model.predictor.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(data_loaders[0]):
            print('Processed set n.: ',i)
            
            # Every data instance is an input + label pair

            # Zero your gradients for every batch!
            #optimizer_low.zero_grad()
            #optimizer_medium.zero_grad()
            #optimizer_high.zero_grad()
            #optimizer_huge.zero_grad()
            optimizer.zero_grad()

            # Make predictions for this batch
            losses = model(data,cfg.modality['target'])
            #print(losses)

           
            loss,log_vars = parse_loss(losses)

            
            


            # Compute the loss gradients
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            #optimizer_low.step()
            #optimizer_medium.step()
            #optimizer_high.step()
            #optimizer_huge.step()
            print('loss this triple',loss.item())
            

            # Gather data and report
            running_loss += losses['low'].item()
            if i % 10 == 9:
                last_loss = running_loss / 10
                print('loss after set n.: ',i, ' value : ',last_loss)
                 # loss per batch
    

        return last_loss

        
    
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 0

    

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

    
        avg_loss = train_one_epoch(epoch_number, writer)
        print('avg_loss',avg_loss)


        epoch_number += 1

    


    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loaders[0]):
            print('TEST Processed set n.: ',i)

    
            #lunch ps only on the target frame


            pre_out,gt_out = model(data,cfg.modality['target'],return_loss = False)
   
            #print('targets',data['id'])
            #print('data',data)
            print(datasets[0].get_ann_info(data['id'][3])['filename_complete'])
            path_target_image = datasets[0].get_ann_info(data['id'][3])['filename_complete']






    






'''
train_image = next(iter_data)
#print('train_image',train_image)

predictions_loss = model(train_image,cfg.modality['target'])
print(predictions_loss)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
'''






 




if __name__ == '__main__':
    main()
