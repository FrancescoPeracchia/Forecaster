import re
from unittest import result
import numpy as np
from .panoptic import save_panoptic_preditcted_gt
import mmcv
import torch
from .utils_train import inference_detector
import os
import json
import shutil
from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import pq_compute_multi_core, \
                                                                      average_pq

    
def evaluate(cfg, model, data_loaders_val,datasets_val):

    print('PANOPTIC SCORE GENERATION')
    data_loader = data_loaders_val[0]
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    



    with torch.no_grad():
        result_GT =[]
        result_PRE = []
        for i, data in enumerate(data_loaders_val[0]):
            

            #features
             
            forecasted_feature,gt_features = model(data,cfg.modality['target'],return_loss = False)

        
            img_info = datasets_val[0].get_ann_info(data['id'][3])['img_info']
            path_target_image = datasets_val[0].get_ann_info(data['id'][3])['filename_complete']
            imgName = img_info['filename']
    
            #features from predictor
            features = [forecasted_feature,gt_features]
            

            
     

            result = [inference_detector(model.efficientps,path_target_image,features[feature], eval='panoptic',) for feature in range(len(features))]
            predicted_ps_forecasting = result[0]
            predictet_ps_gt = result[1]

            result_PRE.append(predicted_ps_forecasting)
            result_GT.append(predictet_ps_gt)

   
          
            save_panoptic_preditcted_gt(predicted_ps_forecasting,predictet_ps_gt)

            #to check this call
            batch_size = 1

            for _ in range(batch_size):
                prog_bar.update()

            

            #print('FORECASTED PS',predicted_ps_forecasting)
            #print('GROUND TRUTH PS',predictet_ps_gt)
            
        merge_gt('tmpDir_GT/tmp_json','')
        eval_results = _evaluate_panoptic()
        print('eval_results',eval_results)

        
        




def _evaluate_panoptic():
    with open('gt.json', 'r') as f:
        gt_json = json.load(f)

    categories = {el['id']: el for el in gt_json['categories']}



    pred_folder = 'tmpDir_PRE/tmp'
    pred_json = 'tmpDir_PRE/tmp_json'
    gt_folder = 'tmpDir_GT/tmp'
    
    
    assert os.path.isdir(gt_folder)
    assert os.path.isdir(pred_folder)
    
    pred_annotations = {}  
    for pred_ann in os.listdir(pred_json):
        with open(os.path.join(pred_json, pred_ann), 'r') as f:
            tmp_json = json.load(f)

        pred_annotations.update({el['image_id']: el for el in tmp_json['annotations']})

    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))


    pq_stat = pq_compute_multi_core(matched_annotations_list, 
                gt_folder, pred_folder, categories)

    results = average_pq(pq_stat, categories)

    metrics = ["All", "Things", "Stuff"]
    msg = "{:14s}| {:>5s}  {:>5s}  {:>5s}".format("Category", "PQ", "SQ", "RQ")
    print('1',msg)
    

    labels = sorted(results['per_class'].keys())
    for label in labels:
        msg = "{:14s}| {:5.1f}  {:5.1f}  {:5.1f}".format(
            categories[label]['name'],
            100 * results['per_class'][label]['pq'],
            100 * results['per_class'][label]['sq'],
            100 * results['per_class'][label]['rq']
        )
    
        print(msg)    
    msg = "-" * 41
    print(msg)
    msg = "{:14s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N")
    print(msg)
    
    eval_results = {} 
    for name in metrics:
        msg = "{:14s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n']
        )

        print(msg)
        eval_results[name+'_pq'] = 100 * results[name]['pq']
        eval_results[name+'_sq'] = 100 * results[name]['sq']
        eval_results[name+'_rq'] = 100 * results[name]['rq']
    


    
    #shutil.rmtree('tmpDir_GT')
    #shutil.rmtree('tmpDir_PRE')
    return eval_results


def merge_gt(path,output_path):

    images = []
    annotations = []




    for json_file in os.listdir(path):
        r = open(os.path.join(path,json_file))
        data_r = json.load(r)
        print('data',data_r)
        print(type(data_r['annotations']))
        print(type(data_r['annotations'][0]))

        annotations.append(data_r['annotations'][0])
        images.append(data_r['images'][0])


    
    c = open('categories.json')
    data_c = json.load(c)
    categories = data_c['categories']

    
    d = {'images': images,
        'annotations': annotations,
        'categories': categories}

    file_name = 'gt'
    

    with open(os.path.join(output_path,file_name+'.json'), 'w') as f:
                json.dump(d, f, sort_keys=True, indent=4)