import torch
import torch.nn.functional as F
from torch import nn

from .baseline import Baseline
from .graph_net import Graph_Net

import torch
from torch import nn
import torch.nn.functional as F
from .baseline import Baseline
from .graph_net import Graph_Net



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):

    def __init__(self, cfg, n_classes, is_depression = False, is_face = True, is_context = True, mode = 'b'):
        super(Net, self).__init__()
        
        self.is_face = is_face
        self.is_context = is_context

        if self.is_face:
            self.f_model = Baseline(cfg)
            self.out_planes = self.f_model.in_planes

        
        if self.is_context:
            self.c_model = Baseline(cfg)
            self.out_planes = self.c_model.in_planes

            
        self.f_graph_module = Graph_Net(self.out_planes, mode)
        self.c_graph_module = Graph_Net(self.out_planes, mode)

        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(self.out_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.is_depression = is_depression
        if self.is_depression:
            self.regressor = nn.Sequential(
                nn.Linear(self.out_planes, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, n_classes + 1)
            )
            self.regressor.apply(weights_init_kaiming)
        else:
            self.weight  = nn.Parameter(torch.FloatTensor(self.out_planes, n_classes))
            # nn.init.xavier_uniform_(self.weight)
            nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        self.p_m = 0.75
        self.n_m = 1 - self.p_m
        self.s = 20
        self.metric_learning = DensityLoss(cfg)

    def forward(self, face, context, label = None, pair_label = None, is_depression = False):
        
        b = face.shape[0]
        
        # context & face graph
        if self.is_face and self.is_context:
            f_feat = self.f_model(face)
            c_feat = self.c_model(context)
        else:
            if self.is_face:
                f_feat = self.f_model(face)
                c_feat = f_feat
            else:
                c_feat = self.c_model(context)
                f_feat = c_feat
        
        f_graph_feat = self.gap(self.f_graph_module(f_feat, c_feat)).view(b, -1)
        c_graph_feat = self.gap(self.c_graph_module(c_feat, f_feat)).view(b, -1)

        feat         = torch.cat([f_graph_feat, c_graph_feat], dim=1)
        graph_feat   = self.bottleneck(feat)

        
        if is_depression == True:
            output = self.regressor(graph_feat)
            return output
        else:
            output = torch.mm(F.normalize(graph_feat, dim=-1), 
                              F.normalize(self.weight, dim=0)).clamp(min=-1, max=1.)

            if self.training and label is not None:

                one_hot = torch.zeros_like(output)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                output = one_hot * (output - 0.05) + (1 - one_hot) * output

                pos_loss = torch.relu(self.p_m - output[one_hot.bool()].flatten())
                neg_loss = torch.relu(output[(1 - one_hot).bool()].flatten() - self.n_m)
                # print(label)
                # print(output.shape)
                # assert i == -1
                restrict_loss = torch.cat([pos_loss, neg_loss], dim=0).norm(p=2) / b
                if pair_label is not None:
                    density_loss = self.metric_learning(graph_feat, pair_label)
                    return output * self.s, restrict_loss + density_loss
                else:
                    return output * self.s, restrict_loss
            else:
                return output * self.s

class DensityLoss(nn.Module):
    def __init__(self, cfg):
        super(DensityLoss, self).__init__()

        self.k = cfg.DATALOADER.NUM_INSTANCE
        self.p_m = 0.75
        self.n_m = 1 - self.p_m
        self.margin = self.p_m - self.n_m

    def forward(self, feats, labels):
        
        feats = F.normalize(feats, dim=-1, p=2)
        N = feats.size(0)
        aff = torch.matmul(feats, torch.t(feats))
        # masking
        id_mask = torch.eye(N).cuda()
        is_pos = (labels.expand(N, N).eq(labels.expand(N, N).t()).float() - id_mask).bool()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        # pairwise constraint
        pos_val = aff[is_pos].contiguous().view(N, -1)
        neg_val = aff[is_neg].contiguous().view(N, -1)

        # optimal difference
        optim_pos = F.relu(self.p_m - pos_val)
        optim_neg = F.relu(neg_val - self.n_m)
        
        # define density
        mean_pos = pos_val.mean(dim=-1, keepdim=True)
        mean_pos = torch.where(mean_pos < self.n_m, torch.ones_like(mean_pos) * self.n_m, mean_pos)

        mean_neg = neg_val.mean(dim=-1, keepdim=True)
        mean_neg = torch.where(mean_neg > self.p_m, torch.ones_like(mean_neg) * self.p_m, mean_neg)
        
        # weighting
        p_w = F.relu(mean_pos - pos_val)
        n_w = F.relu(neg_val - mean_neg)

        pos_loss = (torch.exp(p_w).detach() * optim_pos).norm(p=2) / N
        neg_loss = (torch.exp(n_w).detach() * optim_neg).norm(p=2) / N
        
        return pos_loss + neg_loss