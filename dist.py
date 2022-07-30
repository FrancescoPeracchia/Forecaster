
import os
from datetime import datetime
import argparse
from mmcv import Config
from PS.mmdet.apis import set_random_seed
import torch.multiprocessing as mp
import torch.distributed as dist
from models.model import build_forecaster
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PS.mmdet import __version__
from dataset import build_dataset,build_dataloader
from utils.utils_train import train_one_epoch, validation, get_lr
from utils.utils  import test_one_batch
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR

# NO DISTRIBUTED 
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('output', help='output path for saving the model')
    parser.add_argument('--modality', type=bool, default=True, help='Are available only two modalities True = "Training"\
    #\ and "Model Test", the last one is executing only one batch to test the model')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args

#DISTRIBUTED LUNCHER UTILS
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_multiprocessing(multi_fn,args):
    mp.spawn(multi_fn,
             args=(args,),
             nprocs=args.process_for_node,
             join=True)

def cleanup():
    dist.destroy_process_group()

#DISTRIBUTED TRAINING
def training_fn(rank, args):

    #_____________________________________________________________________________
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, args.world_size)
    world_size = args.world_size  
    cuda_name_PREFIX = 'cuda:'
    cuda = cuda_name_PREFIX+str(rank)
    print('CUDA : ',cuda,' RANK :',rank, ' over world size : ',str(args.world_size))

    #_____________________________________________________________________________

    directory = os.getcwd()
    print('Current directory : ',directory)
    args = parse_args()
    output_path = args.output
    model_testing = args.modality
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
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, device = cuda)
    torch.cuda.set_device(cuda)
    model.cuda(cuda)
    

    


    
    
    #Update the relative path 
    cfg.data.train['ann_file'] = str(directory+cfg.data.train['ann_file'])
    print('ann_file path :', cfg.data.train['ann_file'])

    cfg.data.train['data_root'] = str(directory+cfg.data.train['data_root'])
    print('data_root path :', cfg.data.train['data_root'])

    cfg.data.validation['ann_file'] = str(directory+cfg.data.validation['ann_file'])
    print('ann_file path :', cfg.data.validation['ann_file'])

    cfg.data.validation['data_root'] = str(directory+cfg.data.validation['data_root'])
    print('data_root path :', cfg.data.validation['data_root'])
    

   

    #lists of datesets
    datasets = [build_dataset(cfg.data.train)]



    #lists of dataloaders
    data_loaders = [
        build_dataloader(
            ds,
            cfg.modality['frame_sequence'],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist = True,
            world_size = world_size,
            rank = rank,
            seed=cfg.seed) for ds in datasets
    ]


    model.predictor.train()
    model.efficientps.eval()
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    if rank == 0 :
        summary(model_ddp)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/MODEL_1_TRAINING{}'.format(timestamp))
        output_path = os.path.join(output_path,'model_predictor.pth')

    optimizer = torch.optim.SGD(model_ddp.parameters(), lr=0.5, momentum=0.9)     
    scheduler0 = ExponentialLR(optimizer, gamma=0.9)
    scheduler1 = MultiStepLR(optimizer, milestones=[10,25,45], gamma=3)

 

    best_avg_loss_val = 9999
    epoch_number = 0
    EPOCHS = 70
    skip = False


    if model_testing != True  :
        
        avg_loss,loss_dict = test_one_batch(cfg, model_ddp, data_loaders, optimizer)
        print('avg_loss',avg_loss,' rank :',rank)
    
    else :

        for epoch in range(EPOCHS):
            
            if rank == 0 :
                print('EPOCH {}:'.format(epoch_number + 1))
   
            

            if skip != True :
                
                #YOU SHOULDN'T CALL model.train() model contains both efficientPS and Forecaster 
                #it ll generatate a model with efficientPS parts for training activated
                print('TRAINING EPOCS rank',rank)
                avg_loss,loss_dict = train_one_epoch(cfg, model_ddp, data_loaders, optimizer,rank)
               


        

                #TENSORBOARD GRAPH TRAIN
                if rank == 0 :
                    
                    print('avg_loss : ',avg_loss,' rank :',rank)
                    for log in loss_dict:
                        mean = np.mean(loss_dict[log])
                        print('loss_dict value ',mean)
                        t= os.path.join('Loss',str(log))
                        writer.add_scalar(t,mean,epoch)
            




            #SCHEDULER & TENSORBOARD GRAPH LR
            
            
            scheduler0.step()
            scheduler1.step()
            epoch_number += 1
            if rank == 0 :
                print('\nSaving model\n')
                lr = get_lr(optimizer)
                writer.add_scalar('learning rate',lr,epoch)
                if avg_loss < best_avg_loss_val:
                    best_avg_loss_val = avg_loss
                    torch.save(model_ddp.state_dict(),output_path)
            
        cleanup()
            
   

def main():
    #dist.destroy_process_group()
    
   
    args = parse_args()

    n_gpus = torch.cuda.device_count()
    #n_gpus = 8
    args.n_gpus = n_gpus

    
    n_nodes = 1
    #n_nodes = 2
    args.n_nodes = n_nodes

    #a GPU cointains 2 processes
    gpusXprocess = 1

    args.gpusXprocess = gpusXprocess
    assert n_gpus/gpusXprocess >= 2, f"Requires at least gpus/gpu_for_process = 2 to run, but got {n_gpus/gpusXprocess}"
    
    #Based on the number of nodes and gpus per node, we can calculate the world_size,
    #or the total number of processes to run, which is equal to the total number of gpus/gpu_for_process
    #because we’re assigning multiple-gpu to every process. 
    world_size = int(n_gpus*n_nodes/gpusXprocess)
    args.world_size = world_size

    #Remember, we run the main() function on each node, so that in total there will be n_gpus*n_nodes/gpusXprocess processes
    #but only n_gpus/gpusXprocess processes are on this node (and main() iteration)
    process_for_node = int(n_gpus/gpusXprocess)
    args.process_for_node = process_for_node

    #note: if only 1 node is available world_size = process_for_node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'

    #Now, instead of running the train function once it'll be lunched multiple times on this node 
    #note: training_fn(i, args) receives : i, args as arguments, where i goes from 0 to args.gpus - 1. 
    run_multiprocessing(training_fn,args)
    

if __name__ == '__main__':
    main()