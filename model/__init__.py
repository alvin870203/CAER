from .baseline import Baseline

def build_model(cfg):
    model = Baseline(cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_CHOICE)
    return model
