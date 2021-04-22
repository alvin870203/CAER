import torch

def collate_fn_emotion(batch):
    c_imgs, f_imgs, eids = zip(*batch)
    eids = torch.tensor(eids, dtype=torch.int64)
    return torch.stack(c_imgs, dim=0), torch.stack(f_imgs, dim=0), eids

def collate_fn_depression(batch):
    c_imgs, f_imgs, d_vals = zip(*batch)
    d_vals   = torch.tensor(d_vals, dtype=torch.int64)
    return torch.stack(c_imgs, dim=0), torch.stack(f_imgs, dim=0), d_vals