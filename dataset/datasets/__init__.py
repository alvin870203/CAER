from .CAER_S import CAER_S
from .CAER import CAER
from .AVEC import AVEC
from .dataset_loader import ImageDataset, VideoDataset, DepressionVideoDataset

__factory = {
    'CAER_S': CAER_S,
    'CAER': CAER,
    'AVEC':   AVEC
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
