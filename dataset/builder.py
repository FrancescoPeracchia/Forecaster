from .RawDataset import RawDataset

def build_dataset(cfg, default_args=None):


    return RawDataset(cfg['ann_file'],cfg['pipeline'],cfg['data_root'],cfg['target'],default_args)