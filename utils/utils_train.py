from collections import OrderedDict
import torch
import numpy as np
from PS.mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def parse_loss(losses):
    log_vars = OrderedDict()
    for loss in losses.items():
        #loss is a tuple
        loss_value = loss[1]
        loss_name = loss[0]
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value.to('cuda:0')  for _key, _value in log_vars.items())
  

    log_vars['loss'] = loss

    
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

def inference_detector(model,img,feature,eval=None):
    #img is a path
    cfg = model.cfg
    device = next(model.parameters()).device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    #add our features
    data['features'] = feature
    

    # forward the model
    with torch.no_grad():
        if eval is not None:
            data['eval'] = eval 
        model = model.to('cuda:0')
        #data = data.to('cuda:0')
        
        #print(' inference data',data)

        result = model(return_loss=False,forecasting =True,rescale=True, **data)
    return result

def validation(cfg, model,data_loaders_val):
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


        
        #last processed log is 'loss' the one that we want to append for each prediction 
        #in the dataloader and then mean
        lasts_loss = np.append(last_loss,log)
    average_loss = np.mean(lasts_loss)


    return average_loss,log_loss_dict

def train_one_epoch(cfg, model, data_loaders, optimizer):
    """
    Args: cfg model, data_loaders, optimizer

    Return: average_loss, the  comulative loss 'loss' is processed by mean operation, 
    
            log_loss_dict, is a dicitonary where for each prectiond is stored the related resolution loss, mean is NOT yet computed over
            all the predictions 
    
    """
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
        

        for log_var in log_vars:
            log = log_vars[log_var]
            log_loss_dict[log_var] = np.append(log_loss_dict[log_var],log)


        # Compute the loss gradients
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        

        #last processed log is 'loss' the one that we want to append for each prediction 
        #in the dataloader and then mean
        total_loss = loss.detach().clone()
        total_loss = total_loss.to('cpu')
        #print(total_loss)
        lasts_loss = np.append(last_loss,total_loss)

        
    average_loss = np.mean(lasts_loss)

    


    return average_loss,log_loss_dict
