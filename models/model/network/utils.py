import argparse
import os

import mmcv
import torch
import numpy as np
import json
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.core import cityscapes_originalIds

from PIL import Image




def inference(self,out_path,index,predicted_features):

    images = []
    annotations = []
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_base = out_path.split('/')[-1]
    out_base_folder = os.path.join(out_path, out_base)
    out_base_json = out_base_folder + '.json'

    if not os.path.exists(out_base_folder):
        os.mkdir(out_base_folder)

    originalIds = cityscapes_originalIds()

    for i in index:
            result = self.efficientps.inference_detector(os.path.join(path, i),predicted_features, eval='panoptic')
            pan_pred, cat_pred, _ = result[0]
            pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()

            imageId = imgName.replace("_leftImg8bit.png", "")
            inputFileName = imgName
            outputFileName = imgName.replace("_leftImg8bit.png", "_panoptic.png")
            # image entry, id for image is its filename without extension
            images.append({"id": imageId,
                           "width": int(pan_pred.shape[1]),
                           "height": int(pan_pred.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (pan_pred.shape[0], pan_pred.shape[1], 3), dtype=np.uint8
            )

            panPredIds = np.unique(pan_pred)
            segmInfo = []   
            for panPredId in panPredIds:
                if cat_pred[panPredId] == 255:
                    continue
                elif cat_pred[panPredId] <= 10:
                    semanticId = segmentId = originalIds[cat_pred[panPredId]] 
                else:
                    semanticId = originalIds[cat_pred[panPredId]]
                    segmentId = semanticId * 1000 + panPredId 
                
                isCrowd = 0
                categoryId = semanticId

                mask = pan_pred == panPredId
                color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                pan_format[mask] = color

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segmInfo.append({"id": int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': imageId,
                                'file_name': outputFileName,
                                "segments_info": segmInfo})

            Image.fromarray(pan_format).save(os.path.join(out_base_folder, outputFileName))
            prog_bar.update()

    print("\nSaving the json file {}".format(out_base_json))
    d = {'images': images,
         'annotations': annotations,
         'categories': {}}
    with open(out_base_json, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)




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


def inference_detector(self, img, eval=None):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = self.cfg
    device = next(self.parameters()).device  # model device
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
        result = model(return_loss=False, rescale=True, **data)
    return result