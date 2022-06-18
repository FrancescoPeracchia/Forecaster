from mmdet.core import get_classes
import warnings
from mmcv.runner import load_checkpoint
import torch
import mmcv
from mmdet.models import build_detector
from .base import BaseForecaster

class Forecaster(BaseForecaster):

    def __init__(self,
                efficientPS_config,
                efficientPS_checkpoint, 
                multi_forecasting_modality,
                train_cfg,
                test_cfg
                ):
  
        super(Forecaster,self).__init__()

        self.pretrained = None
        self.training_with_p = False
        self.modality = multi_forecasting_modality
       
        

        self.efficientps = self.init_detector(efficientPS_config,efficientPS_checkpoint,self.device)
        self.freeze_ps()



        #self.feature_forecaster = builder.build_feature_forecaster(model_for)
        #self.forward_from_forecasting = True
        self.target_features_list = dict(low=[],medium=[],high=[],huge=[])
        self.features_list = dict(low=[],medium=[],high=[],huge=[])
        self.list_key = {'0':'low','1':'medium','2':'high','3':'huge'}

        #self.init_weights(pretrained=self.pretrained)

    def init_weights(self, pretrained=None):
        
        pass

    def init_detector(self, config, checkpoint=None, device='cuda:0'):
        """Initialize a detector from config file.

        Args:
            config (str or :obj:`mmcv.Config`): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            'but got {}'.format(type(config)))
        config.model.pretrained = None
        model = build_detector(config.model, test_cfg=config.test_cfg)
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint)
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.warn('Class names are not saved in the checkpoint\'s '
                            'meta data, use COCO classes by default.')
                model.CLASSES = get_classes('coco')
        model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model    

    def extract_ps(self,imgs):
        
        """Initialize a detector from config file.

        Args:
            imgs (List of torch tensor) : Each image in general torch.Size([1024, 2048, 3]) triple or more images 
            
        Returns:
             output (Dictionary) : Dictionary contains different keys for each feature resolution, in each resolution is stored a list with all image features for that resolution

        """
          
 
        with torch.no_grad():
            
            image_features = self.efficientps.extract_feats(imgs)
            #is a generator

            output = self.features_list.copy()
            keys = self.list_key.copy()
            for result in next(image_features):
                print('halo',result.shape)

            for index_image in range(len(image_features)):
                for feature in range(len(output)):
                        
                    output[keys[str(feature)]] += image_features[index_image][feature]
                
            return output


    def freeze_ps(self):
        for param in self.efficientps.parameters():
            param.requires_grad = False

    def concatenate(self,features):
        """Used to Transform extracted features from a Dictionary to a tensor used inside the model.

        Args:
            features (Dictionary) : Cointaing a list of tensors for each features type, lenght of each list depends on n.images processed) : Each image in general torch.Size([1024, 2048, 3]) triple or more images 
            
        Returns:
             output (Dictionary) : from Dictionary = {'low' :[tensor11,tensor12....], 'medium' : ......} to tensors where same "detail features" level tensors are condatenated by putting consegutively same level of "featuter"

        """

        for keys in features.keys() : 
            print(keys)



    def forward_train(self,list_image,list_id, targets):
        #imgs is a list of tensors with shape
        #[(hight,width,channels).....]
        losses = dict()
        
    
        
        #number_images = len(imgs)
  
        

 
        features_list = self.extract_ps(list_image)
        input_forecaster = self.concatenate(features_list)

        return features_list
        
        target_features_list = self.extract_ps(target)
    
        #todo-----------------below
        #predict features
        predicted_features = self.feature_forecaster(features_list)
        
        #compute feature forecasting loss 
        loss_feature = self.feature_forecaster.loss(predicted_features,target_features_list)
        losses.update(loss_feature)

        #Panoptic Segmentation output from predicted features and losses
        if self.training_with_ps:
            img_meta = self.efficientps.get_meta()
            output,loss_ps = self.efficientps(predicted_features,img_meta,forward_from_forecasting = self.forward_from_forecasting)
            losses.update(loss_ps)


        return losses
    

  
