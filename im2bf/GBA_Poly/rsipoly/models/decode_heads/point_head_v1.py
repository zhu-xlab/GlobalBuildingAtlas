import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmcv.runner import BaseModule
import torch.nn.functional as F
import pdb
# from utils import scores_to_permutations, permutations_to_polygons

from ..builder import HEADS
from rsidet.models.utils import build_linear_layer

def scores_to_permutations(scores, ignore_thre=0):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        if ignore_thre is not None:
            valid_rows = (scores[b] > ignore_thre).any(axis=1)
            # valid_cols = (scores[b] > 0).any(axis=0)
            valid_scores = scores[b][valid_rows]
            # assert (valid_rows == valid_cols).all()
            r, c = linear_sum_assignment(-scores[b, valid_rows][:, valid_rows])
            r = valid_rows.nonzero()[0][r]
            c = valid_rows.nonzero()[0][c]

        else:
            r, c = linear_sum_assignment(-scores[b])

        perm[b,r,c] = 1
    return torch.tensor(perm)


def permutations_to_polygons(perm, graph, out='torch', ignore_thre=0, min_poly_size=4):
    B, N, N = perm.shape

    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P:
            head = poly[s][-1]

            t = s+1
            while t < P:
                tail = poly[t][0]
                if head == tail:
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly

    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]

        idx = torch.arange(N)[b_diag]

        if idx.shape[0] > 0:
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1).cpu()
            if ignore_thre is not None:
                valid_rows = (b_perm > ignore_thre).any(dim=1)

                first = first[valid_rows]
                second = second[valid_rows]

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if len(p_idx) < min_poly_size + 1:
                    continue

                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].cpu().numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:] * 300 / 320
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                else:
                    print("Indicate a valid output polygon format")
                    exit()
            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch



def MultiLayerPerceptron(channels: list, batch_norm=True):
    n_layers = len(channels)

    layers = []
    for i in range(1, n_layers):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))

        if i < (n_layers - 1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class Attention(nn.Module):

    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.dim = d_model // n_heads
        self.n_heads = n_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.n_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]

        b, d, h, n = query.shape
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / d**.5
        attn = torch.einsum('bhnm,bdhm->bdhn', torch.nn.functional.softmax(scores, dim=-1), value)

        return self.merge(attn.contiguous().view(b, self.dim*self.n_heads, -1))


class AttentionalPropagation(nn.Module):

    def __init__(self, feature_dim: int, n_heads: int):
        super().__init__()
        self.attn = Attention(n_heads, feature_dim)
        self.mlp = MultiLayerPerceptron([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        message = self.attn(x, x, x)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):

    def __init__(self, feature_dim: int, num_layers: int, first_dim):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv1d(first_dim + 2, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True)
        )

        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_layers)])

        self.conv_desc = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_offset = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, 2, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Hardtanh()
        )

    def forward(self, feat, graph):
        # graph = graph.permute(0,2,1)
        feat = torch.cat((feat, graph), dim=1)
        feat = self.conv_init(feat)

        for layer in self.layers:
            feat = feat + layer(feat)

        desc = self.conv_desc(feat)
        offset = self.conv_offset(feat).permute(0,2,1)
        return desc, offset


class ScoreNet(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        n_points = x.shape[-1]

        x = x.unsqueeze(-1)
        x = x.repeat(1,1,1,n_points)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        # x = self.conv1(masked_x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        return x[:,0]


@HEADS.register_module()
class PointHeadV1(BaseModule):

    def __init__(self, descriptor_dim=64, in_dim_angle=64, remove_last=False,
                 out_dim_angle=36, angle_loss_type='standard', loss_angle_weight=1.,
                 permute_loss_weight=1.0, next_point_loss_weight=20.0,
                 point_cls_loss_weight=0.):
        super(PointHeadV1, self).__init__()

        # Default configuration settings
        self.descriptor_dim = descriptor_dim
        self.remove_last = remove_last
        self.loss_angle_weight = loss_angle_weight
        self.next_point_loss_weight = next_point_loss_weight
        self.point_cls_loss_weight = point_cls_loss_weight
        self.out_dim_angle = out_dim_angle
        self.angle_loss_type = angle_loss_type

        self.fc_angle = build_linear_layer(
            dict(type='Linear'),
            # in_features=in_channels + 0 if pos_enc_dim < 0 else pos_enc_dim,
            in_features=in_dim_angle,
            out_features=out_dim_angle)

        self.fc_point_cls = build_linear_layer(
            dict(type='Linear'),
            # in_features=in_channels + 0 if pos_enc_dim < 0 else pos_enc_dim,
            in_features=in_dim_angle,
            out_features=2)

        self.loss_fun = nn.CrossEntropyLoss(reduction='none')
        self.loss_point_fun = nn.SmoothL1Loss(reduction='none')

    def normalize_coordinates(self, graph, ws, input):
        if input == 'global':
            graph = (graph * 2 / ws - 1)
        elif input == 'normalized':
            graph = ((graph + 1) * ws / 2)
            # graph = torch.round(graph).long()
            graph[graph < 0] = 0
            graph[graph >= ws] = ws - 1
        return graph

    def forward_angle(self, points, point_feats, gt_angles):

        pred_angles = self.fc_angle(point_feats)
        angle_mask = (gt_angles.sum(dim=-1) > 0).view(-1)

        if self.angle_loss_type == 'standard':
            loss_angle = self.loss_fun(pred_angles,
                                       gt_angles.argmax(dim=-1))[angle_mask].sum() / (angle_mask.sum() + 1e-8)
        elif self.angle_loss_type == 'two-ways':
            assert self.out_dim_angle % 2 == 0
            loss_angle_1 = self.loss_fun(
                pred_angles[:, :self.out_dim_angle//2],
                gt_angles[:, :self.out_dim_angle//2].argmax(dim=-1)
            )[angle_mask].sum() / (angle_mask.sum() + 1e-8)

            loss_angle_2 = self.loss_fun(
                pred_angles[:, self.out_dim_angle//2:],
                gt_angles[:, self.out_dim_angle//2:].argmax(dim=-1)
            )[angle_mask].sum() / (angle_mask.sum() + 1e-8)

            loss_angle = (loss_angle_1 + loss_angle_2) / 2.

        losses = dict()
        losses['angle_loss'] = loss_angle * self.loss_angle_weight

        return losses, pred_angles

    def forward_point_cls(self, point_feats, gt_inds):
        pos_mask = gt_inds >= 0
        point_cls_preds = self.fc_point_cls(point_feats)
        loss_point_cls = self.loss_fun(point_cls_preds, pos_mask.to(torch.uint8))
        losses = dict()
        # losses['point_cls_loss'] = loss_point_cls.sum() / (pos_mask.sum() + 1e-8) * self.point_cls_loss_weight
        losses['point_cls_loss'] = loss_point_cls.mean() * self.point_cls_loss_weight
        return losses, point_cls_preds

    def forward_test_angle(self, point_feats):
        return self.fc_angle(point_feats)

    def forward_test_point_cls(self, point_feats):
        return self.fc_point_cls(point_feats)

    def forward(self, return_loss=True, mode='angle', **kwargs):
        if return_loss:
            if mode == 'angle':
                return self.forward_angle(**kwargs)
            elif mode == 'point_cls':
                return self.forward_point_cls(**kwargs)
        else:
            if mode == 'angle':
                return self.forward_test_angle(**kwargs)
            elif mode == 'point_cls':
                return self.forward_test_point_cls(**kwargs)


