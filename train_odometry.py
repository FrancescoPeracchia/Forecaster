from models.OD.model import LSTM as OdometryModel
from models.OD.dataset import OdometryDataset
import os
import torch
import torch.nn as nn
from torch import optim
import argparse
from torch.utils.data import DataLoader
import mmcv
import numpy as np
import matplotlib.pyplot as plt

def parse_args():

    parser = argparse.ArgumentParser(description="Train to odometry model")
    parser.add_argument('data_path', help='train data path')
    parser.add_argument('output', help='output path for saving the model')
    args = parser.parse_args()


    return args



def main():

    args = parse_args()

    data_path = args.data_path
    output = args.output
    output_path_vel = os.path.join(output,'model_vel.pth')
    output_path_yaw = os.path.join(output,'model_yaw.pth')

    layers = 1
    hidden = 100
    features = 1

    model_vel = OdometryModel (features,hidden,layers)
    model_yaw = OdometryModel (features,hidden,layers)
    #input = torch.randn(batch_size, sequence_lengh,features )


    target_data = ['vf','yaw']
    #target_data = ['vf']


    tl = len(target_data)


    dataset = OdometryDataset(data_path,target_data)
    training_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    

    loss_function = nn.MSELoss()

    optimizer_vel = optim.SGD(model_vel.parameters(), lr=0.001)
    optimizer_yaw = optim.SGD(model_yaw.parameters(), lr=0.001)



    MAX = 20
    epochs = range(0,MAX)

    loss_YAW = []
    loss_VEL = []

    
    for epoch in epochs:

        loss_epoch = [0,0]
        loss_epoch_last = [0,0]
        counter = 0
        print(' EPOCHS :', epoch,'/',MAX)





        prog_bar = mmcv.ProgressBar(len(training_loader))
        for data in training_loader:

            
            validity = data[1]
            signals = data[0]
            input_vel = signals[:,:,0]
            input_yaw = signals[:,:,1]

            input_vel = torch.unsqueeze(input_vel,2)      
            input_yaw = torch.unsqueeze(input_yaw,2)      

            #print(signals.shape)
            #print(input_vel.shape)
            #print(input_yaw.shape)


            

            for i in validity:

                if i.item() == False:
            
                    #print('skipped')
                    #print(signals)
                    break
            
                else:

                    model_vel.zero_grad()
                    model_yaw.zero_grad()
                    
                    output_vel,gt_vel,last_seen_vel = model_vel(input_vel)
                    output_yaw,gt_yaw,last_seen_yaw = model_yaw(input_yaw)




                    loss_vel = loss_function(output_vel, gt_vel)
                    loss_yaw = loss_function(output_yaw, gt_yaw)


                    loss_last_vel = loss_function(last_seen_vel, gt_vel)
                    loss_last_yaw = loss_function(last_seen_yaw, gt_yaw)

                    loss_vel.backward()
                    loss_yaw.backward()


                    loss_epoch[0] += loss_vel.item()
                    loss_epoch[1] += loss_yaw.item()

                    loss_epoch_last[0] += loss_last_vel.item()
                    loss_epoch_last[1] += loss_last_yaw.item()
                    counter += 1


                    optimizer_vel.step()
                    optimizer_yaw.step()
                    prog_bar.update()
            

        print('VEL loss ', loss_epoch[0]/counter)
        loss_VEL.append(loss_epoch[0]/counter)
        print('VEL loss_epoch_last', loss_epoch_last[0]/counter)

        print('YAW loss ', loss_epoch[1]/counter)
        loss_YAW.append(loss_epoch[1]/counter)
        print('YAW loss_epoch_last', loss_epoch_last[1]/counter)

    torch.save(model_vel.state_dict(),output_path_vel)
    torch.save(model_yaw.state_dict(),output_path_yaw)

    loss_VEL = np.array(loss_VEL, dtype=np.float32)
    plt.plot(loss_VEL,color = (0,0,1) )
    plt.ylabel('vel (m/s)')
    plt.savefig('vel_loss.png')

    loss_YAW = np.array(loss_YAW, dtype=np.float32)
    plt.plot(loss_YAW,color = (0,1,0) )
    plt.ylabel('yaw (rad)')
    plt.savefig('yaw_loss.png')
        



    yaw_pre = []
    yaw_gt = []
    yaw_last_seen = []


    vel_pre = []
    vel_gt = []
    vel_last_seen = []

    model_vel.eval()
    model_yaw.eval()
    print('EVAL')
    with torch.no_grad():
        prog_bar = mmcv.ProgressBar(len(test_loader))
        print('EVAL')
        for index, data in enumerate(test_loader):
            print('EVAL')
            print(index)

            if index < 500: 
                validity = data[1]
                signals = data[0]
                input_vel = signals[:,:,0]
                input_yaw = signals[:,:,1]

                input_vel = torch.unsqueeze(input_vel,2)      
                input_yaw = torch.unsqueeze(input_yaw,2)      


                for i in validity:

                    if i.item() == False:
                
                        break
                
                    else:

                        
                        output_vel,gt_vel,last_seen_vel = model_vel(input_vel)
                        vel_pre.append(output_vel)
                        vel_gt.append(gt_vel)
                        vel_last_seen.append(last_seen_vel)

                        output_yaw,gt_yaw,last_seen_yaw = model_yaw(input_yaw)
                        yaw_pre.append(output_yaw)
                        yaw_gt.append(gt_yaw)
                        yaw_last_seen.append(last_seen_yaw)
                        prog_bar.update()


        vel_pre = np.array(vel_pre, dtype=np.float32)
        plt.plot(vel_pre,color = (0,0,1) )
        vel_gt = np.array(vel_gt, dtype=np.float32)
        plt.plot(vel_gt,color = (0,1,0) )
        vel_last_seen = np.array(vel_last_seen, dtype=np.float32)
        plt.plot(vel_last_seen,color = (1,0,0) )
        plt.ylabel('vel (m/s)')
        plt.savefig('vel_pred.png')







if __name__ == '__main__':
    main()