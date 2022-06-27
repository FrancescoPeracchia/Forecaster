from __future__ import division
import argparse
import copy
import warnings
from xmlrpc.client import boolean
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
    parser.add_argument('--type', default='Train', help='allows you to decide what to do Train/Test/All')

    args = parser.parse_args()


    return args


def main():
    directory = os.getcwd()

    print(directory)
    args = parse_args()

    output_path = args.output
    GT = args.gt
    MODALITY = args.type


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

    cfg.data.validation['ann_file'] = str(directory+cfg.data.validation['ann_file'])
    print('ann_file path :', cfg.data.validation['ann_file'])

    cfg.data.validation['data_root'] = str(directory+cfg.data.validation['data_root'])
    print('data_root path :', cfg.data.validation['data_root'])

    cfg.data.test['ann_file'] = str(directory+cfg.data.test['ann_file'])
    print('ann_file path :', cfg.data.test['ann_file'])

    cfg.data.test['data_root'] = str(directory+cfg.data.test['data_root'])
    print('data_root path :', cfg.data.test['data_root'])

    #lists of datesets
    datasets = [build_dataset(cfg.data.train)]
    datasets_val = [build_dataset(cfg.data.validation)]
    datasets_test = [build_dataset(cfg.data.test)]


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
    '''
    data_loaders_test = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            seed=cfg.seed) for ds in datasets_test
    ]
    '''

    #optimizer_low = torch.optim.SGD(model.predictor.f2f_low.parameters(), lr=0.001, momentum=0.9)
    #optimizer_medium = torch.optim.SGD(model.predictor.f2f_medium.parameters(), lr=0.001, momentum=0.9)
    #optimizer_high = torch.optim.SGD(model.predictor.f2f_high.parameters(), lr=0.001, momentum=0.9)
    #optimizer_huge = torch.optim.SGD(model.predictor.f2f_huge.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.SGD(model.predictor.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        last_loss = np.array([])
        res_32 = np.array([])
        res_64 = np.array([])
        res_128 = np.array([])
        res_256 = np.array([])
        res_loss = np.array([])
        log_loss_dict = {'low':res_256,'medium':res_128,'high':res_64,'huge':res_32,'loss':res_loss}
 

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(data_loaders[0]):
            print('Processed set n.: ',i)
            

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            losses = model(data,cfg.modality['target'])
            #print(losses)

            
            loss,log_vars = parse_loss(losses)
            #loss is the sum of all the individual losses 
            #log_vars is Ordered Dictionary
            print('log',log_vars)

            for log_var in log_vars:
                log = log_vars[log_var]
                log_loss_dict[log_var] = np.append(log_loss_dict[log_var],log)


            # Compute the loss gradients
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            #print('loss this triple',loss.item())
        lasts_loss = np.append(last_loss,log)
        average_loss = np.mean(lasts_loss)


        return average_loss,log_loss_dict

    def validation(epoch_index, tb_writer):
        last_loss = np.array([])
        res_32 = np.array([])
        res_64 = np.array([])
        res_128 = np.array([])
        res_256 = np.array([])
        res_loss = np.array([])
        log_loss_dict = {'low':res_256,'medium':res_128,'high':res_64,'huge':res_32,'loss':res_loss}
 

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(data_loaders_val[0]):
            print('Processed set n.: ',i)
            # Make predictions for this batch
            losses = model(data,cfg.modality['target'])
            #print(losses)
            loss,log_vars = parse_loss(losses)
            #loss is the sum of all the individual losses 
            #log_vars is Ordered Dictionary
            print('log',log_vars)

            for log_var in log_vars:
                log = log_vars[log_var]
                log_loss_dict[log_var] = np.append(log_loss_dict[log_var],log)


          
            #print('loss this triple',loss.item())
        lasts_loss = np.append(last_loss,log)
        average_loss = np.mean(lasts_loss)


        return average_loss,log_loss_dict





    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/MODEL_1_TRAINING{}'.format(timestamp))
    
    
    if MODALITY == 'TRAIN' or MODALITY == 'ALL':
        print('TRAINING')
        epoch_number = 0
        EPOCHS = 10

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

        
            #YOU SHOULDN'T CALL model.train() model contains both efficientPS and Forecaster 
            #it ll generatate a model with efficientPS parts for training activated
            model.predictor.train()
            avg_loss,loss_dict = train_one_epoch(epoch_number, writer)
            print('avg_loss',avg_loss)

            for log in loss_dict:
                mean = np.mean(loss_dict[log])
                t= os.path.join('Loss',str(log))
                writer.add_scalar(t,mean,epoch)
            
            

            model.eval()
            #note model.efficientps is already at eval() you could even use model.predictor.eval()
            with torch.no_grad():
                print('VALIDATION ')
                avg_loss_val,loss_dict_val = validation(epoch_number, writer)
                print('avg_loss_val',avg_loss_val)

                for log in loss_dict_val:
                    mean = np.mean(loss_dict_val[log])
                    t= os.path.join('Loss_validation',str(log))
                    writer.add_scalar(t,mean,epoch)
                    


            epoch_number += 1

    
    PATH = '/home/fperacch/Forecaster/saved/model.pth'
    torch.save(model,PATH)

    model.eval()
    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)
    
    
    if MODALITY == 'TEST'or MODALITY == 'ALL':
        print('TEST')
        with torch.no_grad():
            for i, data in enumerate(data_loaders_val[0]):
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




if __name__ == '__main__':
    main()
