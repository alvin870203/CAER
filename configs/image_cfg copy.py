from yacs.config import CfgNode as CN

_C = CN()

_C.DATASETS = CN()
_C.DATASETS.NAMES = ('CAER_S')
_C.DATASETS.ROOT_DIR = ('./data/CAER/')

_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'PK_BATCH'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 10
_C.DATALOADER.IMS_PER_BATCH = 70

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
# _C.MODEL.BACKBONE = 'resnet18'
# _C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.BACKBONE = 'convnet'
_C.MODEL.PRETRAIN_CHOICE = ''
_C.MODEL.PRETRAIN_PATH = './imagenet_weights/' + _C.MODEL.BACKBONE + '.pth'
_C.MODEL.SAVE_WEIGHT_PATH = './model_weights/' + _C.MODEL.BACKBONE + '_' + _C.DATASETS.NAMES
_C.MODEL.SAVE_TRAIN_INFO = 'training_info.txt'

_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [112, 112]
# Size of the image during test
_C.INPUT.SIZE_TEST = [112, 112]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER_NAME = 'Adam'
if _C.MODEL.PRETRAIN_CHOICE == 'imagenet':
    _C.OPTIMIZER.EPOCH = 70
    _C.OPTIMIZER.STEP  = [50]
    _C.OPTIMIZER.GAMMA = 0.1
    _C.OPTIMIZER.LR = 5e-4
else: 
    _C.OPTIMIZER.EPOCH = 300
    _C.OPTIMIZER.STEP  = [40, 250]
    _C.OPTIMIZER.GAMMA = 0.1
    _C.OPTIMIZER.LR = 5e-3
_C.OPTIMIZER.WEIGHT_DECAY = 1e-5
_C.OPTIMIZER.E_MARGIN = 0.5
_C.OPTIMIZER.TEST_PER = 10