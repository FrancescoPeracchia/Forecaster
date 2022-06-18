from .RawDataset import RawDataset

def build_dataset(cfg, default_args=None):


    return RawDataset(cfg['ann_file'],cfg['pipeline'],default_args)