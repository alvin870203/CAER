import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import util
from configs.AVEC_cfg import _C as avec_cfg
from configs.image_cfg import _C as caer_cfg
from dataset import make_data_loader
from dataset.collate_batch import collate_fn_depression
from model.overall_net import Net
import math

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = caer_cfg.MODEL.DEVICE_ID
    device = torch.device(caer_cfg.MODEL.DEVICE)
    # define DataLoader 
    avec_train_loader, avec_test_loader, avec_num_classes, avec_label_template = make_data_loader(avec_cfg, mode = 'softmax', collate_fn = collate_fn_depression)
    caer_train_loader, caer_test_loader, caer_num_classes, caer_label_template = make_data_loader(caer_cfg, mode = 'triplet')
    
    model_mode = 'both'
    graph_mode = 'b'
    is_face = True
    is_context = True
    is_transfer = True
    # define DNN 
    e_model  = Net(caer_cfg, caer_num_classes, is_face=is_face, is_context=is_context, mode=graph_mode).to(device)
    d_model  = Net(avec_cfg, avec_num_classes+1, True, is_face=is_face, is_context=is_context, mode=graph_mode).to(device)
    
    # define optimizer
    e_opt = torch.optim.Adam([{'params': e_model.parameters()}],
                              lr=caer_cfg.OPTIMIZER.LR,
                              weight_decay=caer_cfg.OPTIMIZER.WEIGHT_DECAY)

    d_opt = torch.optim.Adam([{'params': d_model.parameters()}],
                              lr=caer_cfg.OPTIMIZER.LR,
                              weight_decay=caer_cfg.OPTIMIZER.WEIGHT_DECAY)

    # define learning rate scheduler
    e_scheduler = torch.optim.lr_scheduler.MultiStepLR(e_opt, milestones=caer_cfg.OPTIMIZER.STEP, gamma=caer_cfg.OPTIMIZER.GAMMA, last_epoch=-1)
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_opt, milestones=caer_cfg.OPTIMIZER.STEP, gamma=caer_cfg.OPTIMIZER.GAMMA, last_epoch=-1)

    # start training
    save_log = caer_cfg.MODEL.SAVE_WEIGHT_PATH + '_' + model_mode + '_' + graph_mode + '_' + caer_cfg.MODEL.SAVE_TRAIN_INFO
    util.logger(save_log, 'start training!', 'w')

    max_acc = 0
    for epoch in range(caer_cfg.OPTIMIZER.EPOCH):
        
        # itrain_d_loss, itrain_d_acc = run(device, avec_train_loader, d_model, d_opt, True)
        # test_d_mae, test_d_rmse = eval(device, avec_test_loader, d_model, True)
        # print(test_d_mae, test_d_rmse)
        # assert i == -1
        # test_e_acc = eval(device, caer_test_loader, e_model)
        # test_d_acc = eval(device, avec_test_loader, d_model, True)
        # print('epoch: {:3d}, emotion acc: {:.4f}, depression acc: {:.4f}'.format(epoch, test_e_acc, test_d_acc))
        # assert i == -1
        # test_d_mae, test_d_rmse = eval(device, avec_test_loader, d_model, True)
        # assert i == -1
        train_e_loss, train_e_acc = run(device, caer_train_loader, e_model, e_opt)

        # additional train 10 epoch
        train_d_loss = 0
        train_d_acc = 0
        if is_transfer:
            update_ema_variables(d_model, e_model, global_step = epoch, is_face=is_face, is_context = is_context)
        k = 5
        for i in range(k):
            itrain_d_loss, itrain_d_acc = run(device, avec_train_loader, d_model, d_opt, True)
            train_d_loss += itrain_d_loss
            train_d_acc  += itrain_d_acc
        train_d_loss /= k
        train_d_acc  /= k
        print('avg depression loss: {:.4f}, avg depression acc: {:.4f}'.format(train_d_loss, train_d_acc))
        print('avg emotion loss: {:.4f}, avg emotion acc: {:.4f}'.format(train_e_loss, train_e_acc))

        if epoch % caer_cfg.OPTIMIZER.TEST_PER == 0:
            test_e_acc = eval(device, caer_test_loader, e_model)
            test_d_mae, test_d_rmse = eval(device, avec_test_loader, d_model, True)
            print('epoch: {:3d}, emotion acc: {:.4f}, depression mae: {:.4f}, depression rmse: {:.4f}'.format(epoch, test_e_acc, test_d_mae, test_d_rmse))

            info = 'epoch: {:3d} =>  train emotion accuracy: {:.4f}, train depression accuracy: {:.4f} | test emotion acc: {:.4f}, test depression mae: {:.4f}, test depression rmse : {:.4f}'.format(epoch, train_e_acc, train_d_acc, test_e_acc, test_d_mae, test_d_rmse)
            if test_e_acc > max_acc:
                max_acc = test_e_acc
        else:
            info = 'epoch: {:3d} => train emotion accuracy: {:.4f}, train depression accuracy: {:.4f}'.format(epoch, train_e_acc, train_d_acc)
        util.logger(save_log, info)

        e_scheduler.step()
        d_scheduler.step()

    print('best accuracy: {:.4f}'.format(max_acc))
    util.logger(save_log, 'best accuracy: {:.4f}'.format(max_acc), 'a')
    torch.save(e_model.state_dict(),   caer_cfg.MODEL.SAVE_WEIGHT_PATH + '_emodel.pth')
    torch.save(d_model.state_dict(),   caer_cfg.MODEL.SAVE_WEIGHT_PATH + '_dmodel.pth')


def update_ema_variables(model, ema_model, alpha=0.999, global_step = 0, is_face = True, is_context = True):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    if is_face:
        for ema_param, param in zip(ema_model.f_model.parameters(), model.f_model.parameters()):
            ema_param.data.mul(alpha).add(param.data, alpha = 1 - alpha)
    if is_context:
        for ema_param, param in zip(ema_model.c_model.parameters(), model.c_model.parameters()):
            ema_param.data.mul(alpha).add(param.data, alpha = 1 - alpha)


def run(device, loader, model, opt, is_depression = False):

    length = len(loader)

    avg_loss = 0.
    avg_acc  = 0.

    bins = 46
    model.train()
    for batch_idx, batch_data in enumerate(loader):
        
        c_images = batch_data[0].to(device)
        f_images = batch_data[1].to(device)
        labels   = batch_data[2].to(device)
        
        if is_depression:
            b, t, c, h, w = c_images.shape
            c_images = c_images.contiguous().view(b * t, c, h, w)
            f_images = f_images.contiguous().view(b * t, c, h, w)
            pair_labels = None
            output = model(f_images, c_images, labels, None, is_depression = True).view(b, t, -1).mean(dim=1)

            # depression value
            pred_d_bin = output[:, 0:bins]
            pred_d_val = torch.sigmoid(output[:, -1])
            mse_loss = F.mse_loss(pred_d_val, labels / (bins - 1))

            # distributed loss
            lbin_mask = torch.zeros_like(pred_d_bin)
            lbin_mask.scatter_(1, labels.view(-1, 1), 1)
            lbin_mask.scatter_(1, (labels + 1).view(-1, 1), 1)
            lbin_mask = lbin_mask.bool()

            pred_d_bin = -F.log_softmax(pred_d_bin, dim=-1)
            bin_loss = pred_d_bin[lbin_mask].contiguous().view(b, -1).mean(dim=-1).mean()
            
            loss = bin_loss + mse_loss
        else:
            pair_labels = labels
            output, metric_loss = model(f_images, c_images, labels, pair_labels)
            ce_loss = F.cross_entropy(output, labels)
            loss = ce_loss + metric_loss
            
        opt.zero_grad()
        loss.backward()
        opt.step()

        if is_depression:
            acc = mse_loss
            # match_result = pred_idx == labels.unsqueeze(1).expand(-1, k)
            # acc = match_result.float().sum(dim=-1).mean()
        else:
            acc = (output.max(1)[1] == labels).float().mean()
        
        if batch_idx % 50 == 0:
            print('batch idx: {:3d}, loss: {:.4f}, acc: {:.4f}'.format(batch_idx, loss.item(), acc.item()))
        avg_loss += loss.item()
        avg_acc  += acc.item()
        
    avg_loss /= length
    avg_acc /= length
    return avg_loss, avg_acc
        
def eval(device, loader, model, is_depression = False):
    bins = 46

    if is_depression:
        avg_mae = 0.
        avg_rmse = 0.
    else:
        avg_acc  = 0.

    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):

            cimages = batch_data[0].to(device)
            fimages = batch_data[1].to(device)
            labels  = batch_data[2].to(device)
            
            if is_depression:
                b, t, c, h, w = cimages.shape
                cimages = cimages.contiguous().view(b * t, c, h, w)
                fimages = fimages.contiguous().view(b * t, c, h, w)
                count += b
                output  = model(fimages, cimages, is_depression=True)
            else:
                output  = model(fimages, cimages)

            if is_depression:
                output = output.contiguous().view(b, t, -1).mean(dim=1)
                pred = torch.sigmoid(output[:, -1]) * (bins-1)
                
                pred_d_bin = output[:, 0:bins]
                pred_d_bin = F.softmax(pred_d_bin, dim=-1)

                vals, idxs = pred_d_bin.topk(k=1, dim=-1)
                idxs = idxs.contiguous().flatten().clamp(min=0, max=bins-1)
                # weight = vals / vals.sum()
                # reg_val = (weight * idxs).sum(dim=-1)
                # # print(reg_val)
                # reg_val = (reg_val + 0.5).long()
                # print(reg_val)
                # assert i == -1

                pred = ((idxs + pred) / 2 ).clamp(min=0., max=bins-1)
                pred = pred.cpu()
                # pred = []
                # for i, idx in enumerate(idxs):
                #     st  = max(idx-1, 0)
                #     mid = idx
                #     ed  = min(idx + 1, 46)
                #     score = torch.Tensor([output[i, st], output[i, ed], output[i, mid]])
                #     id    = torch.Tensor([st, ed, mid])
                #     w     = F.softmax(score, dim=-1)
                #     pred.append((w * score).sum())

                # pred = torch.FloatTensor(pred)
                avg_mae += (torch.abs(labels.cpu() - pred)).sum().item()
                avg_rmse += (torch.abs((labels.cpu() - pred))**2 ).sum().item()
            else:
                acc = (output.max(1)[1] == labels).float().mean()
                avg_acc  += acc.item()
    
    if is_depression:
        
        avg_mae /= count
        avg_rmse /= count
        avg_rmse = math.sqrt(avg_rmse)
        return avg_mae, avg_rmse
    else:
        avg_acc /= len(loader)
        return avg_acc

if __name__ == "__main__":
    main()
