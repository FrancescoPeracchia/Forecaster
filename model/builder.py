from .network import Forecaster



def build_forecaster(cfg, train_cfg=None, test_cfg=None):
    return Forecaster(cfg.efficientPS_config, cfg.efficientPS_checkpoint, cfg.multi_forecasting_modality, train_cfg, test_cfg)