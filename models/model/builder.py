from .network import Forecaster



def build_forecaster(cfg, train_cfg=None, test_cfg=None, eval = False, device = 'cuda:0'):
    return Forecaster(cfg.efficientPS_config, cfg.efficientPS_checkpoint, cfg.multi_forecasting_modality, cfg.predictor_config, train_cfg, test_cfg, eval, device)