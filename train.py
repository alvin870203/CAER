import os

import torch
import torch.nn.functional as F

import util
from configs.image_cfg import _C as cfg
from dataset import make_data_loader
from model.graph_net import Graph_Net
from model.overall_net import Net


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device(cfg.MODEL.DEVICE)
    # define DataLoader 
    train_loader, test_loader, num_classes, label_template = make_data_loader(cfg, mode = 'triplet')
    
    # define DNN 
    model = Net(cfg, num_classes).to(device)

    # define optimizer
    opt = torch.optim.Adam([{'params': model.parameters()},],
                            lr=cfg.OPTIMIZER.LR,
                            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.OPTIMIZER.STEP, gamma=cfg.OPTIMIZER.GAMMA, last_epoch=-1)
    
    util.logger(cfg.MODEL.SAVE_WEIGHT_PATH + '_' + cfg.MODEL.SAVE_TRAIN_INFO, 'start training!', 'w')

    # run epoch
    max_acc = 0
    for epoch in range(cfg.OPTIMIZER.EPOCH):
        epoch = epoch + 1
        
        train_loss, train_acc = run(device, train_loader, model, opt)
        print('avg_loss: {:.4f}, avg_acc: {:.4f}'.format(train_loss, train_acc))
        if epoch % cfg.OPTIMIZER.TEST_PER == 0:
            test_acc  = eval(device, test_loader, model)
            print('epoch: {:3d}, accuracy: {:.4f}'.format(epoch, test_acc))
            info = 'epoch: {:4d} => train loss: {:.4f}, train accuracy: {:.4f} | test accuracy: {:.4f}'.format(epoch, train_loss, train_acc, test_acc)
            if test_acc > max_acc:
                max_acc = test_acc
        else:
            info = 'epoch: {:4d} => train loss: {:.4f}, train accuracy: {:.4f}'.format(epoch, train_loss, train_acc)
            
        util.logger(cfg.MODEL.SAVE_WEIGHT_PATH + '_' + cfg.MODEL.SAVE_TRAIN_INFO, info)
            
        
        scheduler.step()
    print('best accuracy: {:.4f}'.format(max_acc))
    util.logger(cfg.MODEL.SAVE_WEIGHT_PATH + '_' + cfg.MODEL.SAVE_TRAIN_INFO, 'best accuracy: {:.4f}'.format(max_acc), 'a')

    torch.save(model.state_dict(), cfg.MODEL.SAVE_WEIGHT_PATH+'.pth')

def run(device, loader, model, opt):

    length = len(loader)

    avg_loss = 0.
    avg_acc  = 0.

    model.train()
    for batch_idx, batch_data in enumerate(loader):
        fimages = batch_data[0].to(device)
        cimages = batch_data[1].to(device)
        labels  = batch_data[2].to(device)
        
        opt.zero_grad()
        # output, f_feat, c_feat, restrict_loss = model(fimages, cimages, labels)
        output,  restrict_loss = model(fimages, cimages, labels, labels)
        ce_loss = F.cross_entropy(output, labels)

        loss = ce_loss + restrict_loss
        loss.backward()
        opt.step()

        acc = (output.max(1)[1] == labels).float().mean()

        if batch_idx % 50 == 0:
            print('batch idx: {:3d}, restrict_loss: {:.4f}, ce_loss: {:.4f}, loss: {:.4f}, accuracy: {:.4f}'.format(batch_idx, restrict_loss.item(), ce_loss.item(), loss.item(), acc.item()))
        
        avg_loss += loss.item()
        avg_acc  += acc.item()
        
    avg_loss /= length
    avg_acc /= length
    return avg_loss, avg_acc
        

def eval(device, loader, model):

    length = len(loader)

    avg_acc  = 0.

    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            fimages = batch_data[0].to(device)
            cimages = batch_data[1].to(device)
            labels  = batch_data[2].to(device)
            
            output  = model(fimages, cimages)
            
            acc = (output.max(1)[1] == labels).float().mean()
            avg_acc  += acc.item()
            
    avg_acc /= length
    return avg_acc
        


if __name__ == "__main__":
    main()
