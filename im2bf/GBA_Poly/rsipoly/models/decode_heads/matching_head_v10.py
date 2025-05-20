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


def permutations_to_polygons(perm, graph, out='torch', ignore_thre=0):
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
class OptimalMatchingV10(BaseModule):

    def __init__(self, descriptor_dim=64, in_channels=36, in_dim_angle=64, remove_last=False, return_polygons=False, mask_graph=True,
                 block_comp_mask=False, out_dim_angle=36, single_permute=False, angle_loss_type='standard',
                 loss_angle_weight=1., permute_loss_weight=1.0, next_point_loss_weight=20.0, point_cls_loss_weight=0.):
        super(OptimalMatchingV10, self).__init__()

        # Default configuration settings
        self.descriptor_dim = descriptor_dim
        self.sinkhorn_iterations = 100
        self.attention_layers = 4
        self.correction_radius = 0.05
        self.remove_last = remove_last
        self.return_polygons = return_polygons
        self.mask_graph = mask_graph
        self.block_comp_mask = block_comp_mask
        self.single_permute = single_permute
        self.loss_angle_weight = loss_angle_weight
        self.permute_loss_weight = permute_loss_weight
        self.next_point_loss_weight = next_point_loss_weight
        self.angle_loss_type = angle_loss_type
        self.point_cls_loss_weight = point_cls_loss_weight

        # Modules
        self.scorenet1 = ScoreNet(self.descriptor_dim * 2)
        self.scorenet2 = ScoreNet(self.descriptor_dim * 2)
        # self.gnn = AttentionalGNN(self.descriptor_dim, self.attention_layers)
        self.gnn = AttentionalGNN(self.descriptor_dim, self.attention_layers, first_dim=in_channels)

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

        # self.loss_fun = nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_fun = nn.CrossEntropyLoss(reduction='mean')
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

    def predict(self, image, descriptors, graph):
        B, _, H, W = image.shape
        B, N, _ = graph.shape

        for b in range(B):
            b_desc = descriptors[b]
            b_graph = graph[b]

            # Extract descriptors
            b_desc = b_desc[:, b_graph[:,0], b_graph[:,1]]

            # Concatenate descriptors in batches
            if b == 0:
                sel_desc = b_desc.unsqueeze(0)
            else:
                sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0)

        # Multi-layer Transformer network.
        norm_graph = self.normalize_coordinates(graph, W, input="global") #out: normalized coordinate system [-1, 1]
        sel_desc, offset = self.gnn(sel_desc, norm_graph)

        # Correct points coordinates
        norm_graph = norm_graph + offset * self.correction_radius
        graph = self.normalize_coordinates(norm_graph, W, input="normalized") # out: global coordinate system [0, W]

        # Compute scores
        scores_1 = self.scorenet1(sel_desc)
        scores_2 = self.scorenet2(sel_desc)
        scores = scores_1 + torch.transpose(scores_2, 1, 2)

        scores = scores_to_permutations(scores)
        poly = permutations_to_polygons(scores, graph, out='numpy')

        return poly

    def forward_train(self, point_feats, graph_targets, point_preds, point_preds_ori, comp_mask=None, probs=None):
        losses = {}

        B, N, C = point_feats.shape

        if B == 0 or N == 0:
            return {'permute_loss': torch.zeros(1, device=point_feats.device)}, {}

        gnn_feat, offset = self.gnn(point_feats.permute(0,2,1),
                                    point_preds.permute(0,2,1)) # B, C, N
        offset_preds = point_preds + offset * self.correction_radius

        scores_1 = self.scorenet1(gnn_feat)
        scores_2 = self.scorenet2(gnn_feat)

        if comp_mask is not None and not self.block_comp_mask:
            # scores_1 = torch.where(comp_mask==1, scores_1, -torch.inf)
            # scores_2 = torch.where(comp_mask==1, scores_2, -torch.inf)
            scores_1 = torch.where(comp_mask==1, scores_1, -1e8)
            scores_2 = torch.where(comp_mask==1, scores_2, -1e8)

        gt_permute_1 = graph_targets.argmax(dim=-1)
        gt_permute_2 = graph_targets.transpose(1, 2).argmax(dim=-1)

        if comp_mask is not None and not self.block_comp_mask:
            mask = ~(torch.where(comp_mask == 1, 0, graph_targets) == 1)
            # if (mask == False).sum() > 0:
            #     pdb.set_trace()
            mask_1 = mask.all(dim=-1)
            mask_2 = mask.all(dim=1)
        else:
            mask_1 = torch.ones_like(graph_targets).all(dim=-1)
            mask_2 = torch.ones_like(graph_targets).all(dim=-1)

        if self.mask_graph:
            mask_1 = mask_1 & (graph_targets > 0).any(dim=-1)
            mask_2 = mask_2 & (graph_targets > 0).any(dim=1)

        # loss_gnn_1 = (self.loss_fun(scores_1, gt_permute_1)[mask_1]).sum() / (mask_1.sum() + 1e-8)
        # loss_gnn_2 = (self.loss_fun(scores_2, gt_permute_2)[mask_2]).sum() / (mask_2.sum() + 1e-8)
        loss_gnn_1 = (self.loss_fun(scores_1.view(B*N, -1),
                                    gt_permute_1.view(B*N))[mask_1.view(-1)]).sum() / (mask_1.sum() + 1e-8)
        loss_gnn_2 = (self.loss_fun(scores_2.view(B*N, -1),
                                    gt_permute_2.view(B*N))[mask_2.view(-1)]).sum() / (mask_2.sum() + 1e-8)

        if self.single_permute:
            losses['permute_loss'] = loss_gnn_1 * self.permute_loss_weight
        else:
            losses['permute_loss'] = (loss_gnn_1 + loss_gnn_2) / 2. * self.permute_loss_weight

        probs_1 = F.softmax(scores_1, dim=-1)
        accu_points = torch.einsum('bnn,bnc->bnc', probs_1, point_preds).view(-1, 2)
        next_point_preds = point_preds.view(-1, 2)[gt_permute_1.view(-1)]
        loss_point = self.loss_point_fun(accu_points[mask_1.view(-1)], next_point_preds[mask_1.view(-1)]).sum() / (mask_1.sum() + 1e-8)
        losses['next_point_loss'] = loss_point * self.next_point_loss_weight


        # loss_gnn_1 = (self.loss_fun(scores_1, gt_permute_1) * mask).sum() / (mask.sum() + 1e-8)
        # loss_gnn_2 = (self.loss_fun(scores_2, gt_permute_2) * mask).sum() / (mask.sum() + 1e-8)

        state = {}
        # state['point_preds'] = [point_pred for point_pred in point_preds_ori.detach().cpu()]
        state['point_preds_ori'] = point_preds_ori
        if self.return_polygons:
            # scores = (scores_1 + scores_2.transpose(1, 2)).detach()
            scores =  scores_1

            probs = F.softmax(scores_1, dim=-1)
            prob_permute = scores_to_permutations(probs, ignore_thre=0.0)
            polygons = permutations_to_polygons(prob_permute, point_preds_ori, out='numpy')

            # scores = graph_targets.float()
            # scores = (graph_targets.float() + graph_targets.float().transpose(1, 2))
            # scores = scores_to_permutations(scores)
            # # temp = self.normalize_coordinates(point_preds, img.shape[-1], 'normalized').detach()
            # polygons = permutations_to_polygons(scores, point_preds_ori, out='numpy')

            return_polygons = []
            for polygon_per_batch in polygons:
                return_polygons.extend(polygon_per_batch)
            state['return_polygons'] = [return_polygons]

        return losses, state

    def forward_test(self, img, point_feats, point_preds, comp_mask):
        if self.remove_last:
            point_feats = point_feats[:, :, :-2]

        point_angles = self.fc_angle(point_feats)
        B, N, num_angle = point_angles.shape

        B, _, H, W = img.shape
        assert B == 1
        # gnn_feat, offset = self.gnn(point_feats.permute(0,2,1),
        #                             point_preds.permute(0,2,1)) # B, C, N
        gnn_feat, offset = self.gnn(point_angles.permute(0,2,1),
                                    point_preds.permute(0,2,1)) # B, C, N
        # scores_1 = F.softmax(self.scorenet1(gnn_feat), dim=-1)
        # scores_2 = F.softmax(self.scorenet2(gnn_feat), dim=-1)
        scores_1 = self.scorenet1(gnn_feat)
        scores_2 = self.scorenet2(gnn_feat)
        scores_1 = torch.where(comp_mask==1, scores_1, -torch.inf)
        scores_2 = torch.where(comp_mask==1, scores_2, -torch.inf)
        scores_1 = F.softmax(scores_1, dim=-1)
        scores_2 = F.softmax(scores_2, dim=-1)

        scores = scores_1 + scores_2.transpose(1, 2)
        scores = scores_to_permutations(scores)
        temp = self.normalize_coordinates(point_preds, img.shape[-1], 'normalized').detach()
        polygon = permutations_to_polygons(scores, temp, out='numpy')[0]

        return [polygon], [point_pred for point_pred in temp.detach().cpu()]

    def forward_test_global(self, point_feats, point_preds, point_preds_ori, comp_mask, **kwargs):
        # if self.remove_last:
        #     point_feats = point_feats[:, :, :-2]
        # point_angles = self.fc_angle(point_feats)
        # B, N, num_angle = point_angles.shape

        gnn_feat, offset = self.gnn(point_feats.permute(0,2,1), point_preds.permute(0,2,1)) # B, C, N
        # scores_1 = F.softmax(self.scorenet1(gnn_feat), dim=-1)
        # scores_2 = F.softmax(self.scorenet2(gnn_feat), dim=-1)
        scores_1 = self.scorenet1(gnn_feat)
        scores_2 = self.scorenet2(gnn_feat)
        scores_1 = torch.where(comp_mask==1, scores_1, -1e8)
        scores_2 = torch.where(comp_mask==1, scores_2, -1e8)
        scores_1 = F.softmax(scores_1, dim=-1)
        scores_2 = F.softmax(scores_2, dim=-1)
        pdb.set_trace()

        if self.single_permute:
            scores = scores_1
        else:
            scores = scores_1 + scores_2.transpose(1, 2)

        scores = scores_to_permutations(scores)

        # temp = self.normalize_coordinates(point_preds, img.shape[-1], 'normalized').detach()
        polygon = permutations_to_polygons(scores, point_preds_ori, out='numpy')

        return polygon, [point_pred_ori for point_pred_ori in point_preds_ori.detach().cpu()]

    def forward_angle(self, points, point_feats, gt_angles):

        pred_angles = self.fc_angle(point_feats)
        angle_mask = (gt_angles.sum(dim=-1) > 0).view(-1)

        if self.angle_loss_type == 'standard':
            loss_angle = self.loss_fun(pred_angles,
                                       gt_angles.argmax(dim=-1))[angle_mask].sum() / (angle_mask.sum() + 1e-8)

        losses = dict()
        losses['angle_loss'] = loss_angle * self.loss_angle_weight

        return losses, pred_angles

    def forward_point_cls(self, point_feats, gt_inds):
        pos_mask = gt_inds >= 0
        point_cls_preds = self.fc_point_cls(point_feats)
        loss_point_cls = self.loss_fun(point_cls_preds, pos_mask.to(torch.uint8))
        losses = dict()
        losses['point_cls_loss'] = loss_point_cls[pos_mask].sum() / (pos_mask.sum() + 1e-8) * self.point_cls_loss_weight
        return losses

    def forward_test_angle(self, point_feats):
        return self.fc_angle(point_feats)

    def forward(self, return_loss=True, mode='local', **kwargs):
        if return_loss:
            if mode == 'local':
                return self.forward_train(**kwargs)
            elif mode == 'angle':
                return self.forward_angle(**kwargs)
            elif mode == 'point_cls':
                return self.forward_point_cls(**kwargs)
        else:
            if mode == 'local':
                return self.forward_test(**kwargs)
            elif mode == 'global':
                return self.forward_test_global(**kwargs)
            elif mode == 'angle':
                return self.forward_test_angle(**kwargs)


