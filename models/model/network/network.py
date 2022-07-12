from cmath import e
from operator import gt
from mmdet.core import get_classes
import warnings
from mmcv.runner import load_checkpoint
import torch
import mmcv
from mmdet.models import build_detector
from .base import BaseForecaster
from .F2F import F2F
import copy
import torch.nn as nn

class Forecaster(BaseForecaster):

    def __init__(self,
                efficientPS_config,
                efficientPS_checkpoint, 
                multi_forecasting_modality,
                forecaster_cfg,
                train_cfg,
                test_cfg,
                eval
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
        self.inv_list_key = {'low':'0','medium':'1','high':'2','huge':'3'}
        self.ps_output = dict(current=[],target=[],gt=[],time=[],index=[])
        self.head_input = dict(Predicted=[],GT=[])

        self.predictor = F2F(forecaster_cfg.model)
        

        if eval :
            PATH = forecaster_cfg.weights[0]
            print('Loading pre-trained Forecaster...')
            print('Forecaster Model checkpoint at : ',PATH)

            checkpoint = torch.load(PATH)     
            self.predictor.load_state_dict(checkpoint)
            
            self.predictor.eval()
            self.freeze_forecaster()


        #self.init_weights(pretrained=self.pretrained)

    def init_weights(self, pretrained=None):
        
        pass

    def freeze_ps(self):
        for param in self.efficientps.parameters():
            param.requires_grad = False

    def freeze_forecaster(self):
        for param in self.predictor.parameters():
            param.requires_grad = False


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

    def extract_ps(self,imgs,list_id):
        
        """Initialize a detector from config file.

        Args:
            imgs (List of torch tensor) : Each image in general torch.Size([1024, 2048, 3]) triple or more images 
            
        Returns:
             output (Dictionary) : Dictionary contains different keys for each feature resolution, in each resolution is stored a list with all image features for that resolution

        """


 
        with torch.no_grad():
            
            #from PS code
            image_features = self.efficientps.extract_feats(imgs,list_id)
            #is a generator gives in output a tuple with 4 features resolutions

            #feature_list = self.features_list.copy()
            output = copy.deepcopy(self.features_list)
            #output = dict(feature_list)
            keys = self.list_key.copy()
            
            
            #call the extraction for all the loades imgs, note above is a generator
            for result,id in image_features:
                #print('result',len(result))
                #print('id',id)

              

                for feature in range(len(result)):
                    
                        
                    output[keys[str(feature)]].append(result[feature])
                
            return output


    def prepare_features(self,features):
        
        """Used to Transform extracted features from a Dictionary to a tensor used inside the model.

        Args:
             input (Dictionary) : Dictionary contains different keys for each feature resolution, in each resolution is stored a list with all image features for that resolution
            
        Returns:
             features (Dictionary) : Cointaining two key "Predicted" "GT", but keys referr to a list of same lenght, features compose each element of both lists

        """
        output = self.head_input.copy()
        dim = len(features['low'])
        
        

        
        for i in range(int(dim/2)):
            pre_list_temp = []
            gt_list_temp = []
            for feature in range(len(features)):
                

                feature_predicted = features[self.list_key[str(feature)]][i]
                feature_gt = features[self.list_key[str(feature)]][int(i+(dim/2))]
                pre_list_temp.append(feature_predicted)
                gt_list_temp.append(feature_gt)
            



            output['Predicted'].append(feature_predicted)
            output['GT'].append(gt_list_temp)


        return output

    def extract_results(self,features,list_id,targets):

        """
        Args:
            features (Dictionary) : Cointaining two key "Predicted" "GT", but keys referr to a list of same lenght, features compose each element of both lists
            list_id (List) : Keep track of the image id processed, depends on how the Sample is working
            target (List) : Gives us information on how many steps forward the model il predicting i.e [1,2,3] means that the fist element is one step forward, the second 
            2 and the third 3
        Return:
            output (Dictionary) : Cointaining two key "Predicted" "GT", but keys referr to a list of same lenght with the heads output generated for but GT and Predicted Features
         
        """
  

        output = self.ps_output.copy()
        #print(len(targets))
        #print(len(features['GT']))

        assert len(features['Predicted']) == len(features['GT']), "Predicted and GT features are not consistent, must be equal in number"
        assert len(targets) == len(features['GT']), "Wanted Target frames GT features are not consistent, must be equal in number"
        
        
        for i in range(len(features['Predicted'])):
            predicted_features =  features['Predicted'][i]
            gt_features =  features['GT'][i]
            time = targets[i]
            index = list_id[i]
            


            predicted_ps, gt_ps = self.efficientps.results_from_features(predicted_features,gt_features)
            output['Predicted'].append(predicted_ps)
            output['GT'].append(gt_ps)
            output['time'].append(time)
            output['index'].append(index)
        

        
            return output 
    
    def forward_ps(self,pre,gt,list_id,target):
        
        return


    def forward_train(self,list_image,list_id, targets):
        #imgs is a list of tensors with shape
        #[(hight,width,channels).....]
       
        
        
        features_dict = self.extract_ps(list_image,list_id)
        #print('dept',len(features_list['low']))

        losses = self.predictor(features_dict, targets, return_loss=True)


        return losses
    

    def forward_test(self,list_image,list_id, targets):
        self.loss = nn.L1Loss()
        
        features_dict = self.extract_ps(list_image)
        #return loss is false
        predicted_features,gt_features = self.predictor(features_dict, targets, return_loss=False)

        #check is feature target is still the same
        vera = features_dict['low'][3] 
        #print('vera shape', vera.shape)
        #print('feature dict',len(features_dict['low']))
        gt = gt_features['low']
        #print('gt shape', gt.shape)
        loss_ = self.loss(vera, gt)
        #print('loss',loss_)


        
        return predicted_features,gt_features


    




        
  
