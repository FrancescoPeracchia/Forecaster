from __future__ import division
import argparse
import copy
import warnings
from PS.mmdet.models import efficientps
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from mmcv import Config
from PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from model import build_forecaster
from PS.mmdet.apis import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_train import parse_loss,inference_detector

import os




from PIL import Image
from PS.mmdet.datasets.cityscapes import PALETTE
import numpy as np
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries


#CALL
#Forecaster$ python train.py /home/fperacch/Forecaster/configs/Forecast_multigpu_sample.py /home/fperacch/Forecaster/data/kitti_raw/output False



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('output', help='output path')
    parser.add_argument('gt',type=bool, help='True if you want to produce in output also the gt map\
                        ,False if is already computed and you want to generate only results from predictions')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()


    return args


def main():
    directory = os.getcwd()

    print(directory)
    args = parse_args()

    output_path = args.output
    GT = args.gt

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

    #lists of datesets
    datasets = [build_dataset(cfg.data.train)]
    datasets_val = [build_dataset(cfg.data.train)]
    datasets_test = [build_dataset(cfg.data.train)]


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

    data_loaders_test = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets_test
    ]


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

    
        #YOU SHOULDN'T CALL model.train() model contains both efficientPS and Forecaster 
        #it ll generatate a model with efficientPS parts for training activated
        model.predictor.train()
        avg_loss = train_one_epoch(epoch_number, writer)
        print('avg_loss',avg_loss)
        

        model.eval()
        #note model.efficientps is already at eval() you could even use model.predictor.eval()
        with torch.no_grad():
            print('VALIDATION ')
            for i, data in enumerate(data_loaders_test[0]):


        epoch_number += 1

    

    torch.save(model,'')

    model.eval()
    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)
    
    
    with torch.no_grad():
        for i, data in enumerate(data_loaders_test[0]):
            print('TEST Processed set n.: ',i)


            pre_out,gt_out = model(data,cfg.modality['target'],return_loss = False)

           
            img_info = datasets[0].get_ann_info(data['id'][3])['img_info']
            path_target_image = datasets[0].get_ann_info(data['id'][3])['filename_complete']
            imgName = img_info['filename']
    
     
            prediction_path = os.path.join(output_path,'forecasted')
            list_path =[prediction_path]
            features = [pre_out]
            if GT :
                features.append(gt_out)
                gt_path = os.path.join(output_path,'gt')
                list_path.append(gt_path)
            

            for i,feature in enumerate(features):
                save_path = list_path[i]
                result = inference_detector(model.efficientps,path_target_image,feature, eval='panoptic',)
                pan_pred, cat_pred, _ = result[0]

                
                img = Image.open(path_target_image)
                out_path = os.path.join(save_path, imgName)

                sem = cat_pred[pan_pred].numpy()
                sem_tmp = sem.copy()
                sem_tmp[sem==255] = colors.shape[0] - 1
                sem_img = Image.fromarray(colors[sem_tmp])

                is_background = (sem < 11) | (sem == 255)
                pan_pred = pan_pred.numpy() 
                pan_pred[is_background] = 0

                contours = find_boundaries(pan_pred, mode="outer", background=0).astype(np.uint8) * 255
                contours = dilation(contours)

                contours = np.expand_dims(contours, -1).repeat(4, -1)
                contours_img = Image.fromarray(contours, mode="RGBA")

                out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
                out = Image.alpha_composite(out, contours_img)
                out.convert(mode="RGB").save(out_path)





    






'''
train_image = next(iter_data)
#print('train_image',train_image)

predictions_loss = model(train_image,cfg.modality['target'])
print(predictions_loss)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
'''






 




if __name__ == '__main__':
    main()
