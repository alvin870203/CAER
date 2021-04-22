from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from .collate_batch import collate_fn_emotion
from .datasets import init_dataset, ImageDataset, VideoDataset, DepressionVideoDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def make_data_loader(cfg, mode = 'softmax', collate_fn = collate_fn_emotion):
    train_transforms_f = build_transforms(cfg, is_drop=False, is_train=True, is_face=True)
    train_transforms_c = build_transforms(cfg, is_drop=True,  is_train=True, is_face=False)
    val_transforms_f   = build_transforms(cfg, is_train=False, is_face=True)
    val_transforms_c   = build_transforms(cfg, is_train=False, is_face=False)
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    label_template = dataset.label_template
    num_classes = len(label_template)

    
    if cfg.DATALOADER.DATA_TYPE == 'IMAGE':
        train_set = ImageDataset(cfg, dataset.train, train_transforms_f, train_transforms_c)
        test_set  = ImageDataset(cfg, dataset.test,  val_transforms_f, val_transforms_c)
    else:
        train_set = DepressionVideoDataset(cfg, dataset.train, train_transforms_f, train_transforms_c, mode='train')
        test_set  = DepressionVideoDataset(cfg, dataset.test,  val_transforms_f, val_transforms_c, mode='test')
    
    
    if mode == 'triplet':
        train_loader = DataLoader(
            train_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.DATALOADER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=collate_fn
            )
    elif mode == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, drop_last = True
            )
    else:
        raise ValueError('not implementation', mode)
    
    test_loader = DataLoader(
        test_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, test_loader, num_classes, label_template
