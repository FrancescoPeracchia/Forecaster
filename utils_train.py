from collections import OrderedDict
import torch

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

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
    print('total loss',loss)
    print('total loss',type(loss))

    log_vars['loss'] = loss
    print('log_vars',log_vars)
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars



from PS.mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch.nn as nn
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

def inference_detector(model,img,pre_out,gt_out,eval=None):
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
    # forward the model
    with torch.no_grad():
        if eval is not None:
            data['eval'] = eval 
        model = model.to('cuda:0')
        #data = data.to('cuda:0')
        print(data)
        result = model(return_loss=False, rescale=True, **data)
    return result
