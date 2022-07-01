import torch
from PIL import Image
import os
from .utils_train import inference_detector
import numpy as np
from PIL import Image
from PS.mmdet.datasets.cityscapes import PALETTE
import numpy as np
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

def save_prediction(cfg, model, data_loaders_val,datasets_val):

    print('IMAGE GENERATION')
    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)
    output_path = cfg.test['GT']
    GT = cfg.test['GT']


    with torch.no_grad():
        for i, data in enumerate(data_loaders_val[0]):
            print('TEST Processed set n.: ',i)


            pre_out,gt_out = model(data,cfg.modality['target'],return_loss = False)

        
            img_info = datasets_val[0].get_ann_info(data['id'][3])['img_info']
            path_target_image = datasets_val[0].get_ann_info(data['id'][3])['filename_complete']
            imgName = img_info['filename']
    
    
            prediction_path = os.path.join(output_path,'forecasted')
            list_path =[prediction_path]
            features = [pre_out]
            if  GT :
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