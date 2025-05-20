import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmcv.runner import BaseModule
import torch.nn.functional as F
import pdb
import torch_geometric.nn.models as geom_models
import random
# from utils import scores_to_permutations, permutations_to_polygons

from ..builder import HEADS
from rsidet.models.utils import build_linear_layer
from shapely.geometry import Polygon
from positional_encodings.torch_encodings import PositionalEncoding2D
from .. import builder

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

    def __init__(self, feature_dim: int, num_layers: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv1d(in_channels, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
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
            nn.Conv1d(feature_dim, out_channels, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm1d(feature_dim),
            # nn.ReLU(inplace=True)
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

    def forward(self, feat):
        # graph = graph.permute(0,2,1)
        # feat = torch.cat((feat, graph), dim=1)
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
class DeepQNetV4(BaseModule):

    def __init__(
        self, gnn_net, in_channels=512, hidden_channels=64, num_gnn_layers=3, return_polygons=False,
        max_iter=2000, final_eps=1e-4, init_eps=0.1, extra_poly_penalty=0.2,
        num_limit_points_per_batch=512*512*8, replay_memory_size=32, gamma=0.99,
        num_gnn_hop=5, apply_rev_gnn_feats=True, only_use_state=False, pass_reward_thre=0.8,
        gt_eps=0.5, channels_pos_enc=2, out_gnn_channels=256, add_gt_state=False,
        loss_row_col_weight=100., loss_dqn_weight=1.0, loss_gt_state_weight=10.,
        reward_invalid_0=-1., reward_invalid_1=-0.5, reward_invalid_2=-0.2, reward_valid2=-0.5,
        rand_sample_action=False, early_stop=True, game_pool_size_limit = 256
    ):
        super(DeepQNetV4, self).__init__()

        self.gnn = builder.build_backbone(gnn_net)

        self.conv_desc = nn.Sequential(
            nn.Conv2d(out_gnn_channels, 128, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm1d(feature_dim),
            # nn.ReLU(inplace=True)
        )

        # self.loss_fun = nn.CrossEntropyLoss(reduction='none')
        # self.loss_fun = nn.SmoothL1Loss(reduction='none')
        self.loss_fun = nn.SmoothL1Loss()

        # self.gnn = geom_models.GAT(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     num_layers=num_gnn_layers,
        #     K=num_gnn_hop,
        # )

        self.score_net = ScoreNet(hidden_channels * 2)
        self.max_iter = max_iter
        self.final_eps = final_eps
        self.init_eps = init_eps
        self.extra_poly_penalty = extra_poly_penalty
        self.num_limit_points_per_batch = num_limit_points_per_batch
        self.replay_memory_size = replay_memory_size
        self.gamma = gamma
        self.apply_rev_gnn_feats = apply_rev_gnn_feats
        self.pass_reward_thre = pass_reward_thre
        self.gt_eps = gt_eps
        self.only_use_state = only_use_state
        self.pos_encoding = PositionalEncoding2D(channels_pos_enc)
        self.channels_pos_enc = channels_pos_enc
        self.add_gt_state = add_gt_state
        self.loss_dqn_weight = loss_dqn_weight
        self.loss_row_col_weight = loss_row_col_weight
        self.loss_gt_state_weight = loss_gt_state_weight
        self.reward_invalid_0 = reward_invalid_0
        self.reward_invalid_1 = reward_invalid_1
        self.reward_invalid_2 = reward_invalid_2
        self.reward_valid2 = reward_valid2
        self.rand_sample_action = rand_sample_action
        self.early_stop = early_stop
        self.game_pool = {}
        self.game_pool_size_limit = game_pool_size_limit

    def set_environment(self, point_feats_list, gt_edges_list, point_preds_list,
                        point_preds_ori_list, cur_iter=0):

        self.point_feats_list = [x.detach() for x in point_feats_list]
        # self.point_feats_list = [torch.cat([x1.detach(), x2], dim=1) for x1, x2 in zip(point_feats_list, point_preds_list)]
        self.gt_edges_list = gt_edges_list
        self.point_preds_list = [x.detach() for x in point_preds_list]
        self.point_preds_ori_list = point_preds_ori_list
        self.gt_polygons_list = [None] * len(self.gt_edges_list)
        self.cur_iter = cur_iter

    def init_game(self, skill_level):

        sizes = np.array([len(x) for x in self.point_feats_list])
        # valid_idx = ((sizes <= skill_level) & (sizes > skill_level-3)).nonzero()[0]
        # if len(valid_idx) == 0:
        #     return False, -1

        # self.game_idx = random.choice(valid_idx)
        # N, _ = self.point_feats_list[self.game_idx].shape
        # self.cur_state = self.first_state.clone()
        self.replay_memory = []
        self.cnt_invalid_0 = 0
        self.cnt_invalid_1 = 0
        self.cnt_invalid_2 = 0
        self.cnt_valid = 0
        self.iou_scores = []
        self.cnt_valid_2 = 0
        self.cnt = 0
        # self.gt_edges = self.gt_edges_list[self.game_idx]
        # if skill_level - 3 in self.game_pool.keys():
        #     del(self.game_pool[skill_level-3])

        for level in range(skill_level - 2, skill_level+1):
            if not level in self.game_pool.keys():
                self.game_pool[level] = []

            valid_idx = (sizes == level).nonzero()[0]
            for idx in valid_idx:
                point_preds_ori = self.point_preds_ori_list[idx]
                edges_list = self.gt_edges_list[idx].tolist()
                poly_idxes_list = self.state2polygons(edges_list)
                gt_polygons = [Polygon(point_preds_ori[poly_idxes].tolist()) for poly_idxes in poly_idxes_list]

                game = dict(
                    point_feats = self.point_feats_list[idx],
                    gt_edges = self.gt_edges_list[idx],
                    point_preds = self.point_preds_list[idx],
                    point_preds_ori = self.point_preds_ori_list[idx],
                    gt_polygons = gt_polygons,
                    first_state = torch.zeros(
                        level, level,
                        device=self.point_feats_list[idx].device,
                        dtype=torch.int
                    ),
                    cur_state = torch.zeros(
                        level, level,
                        device=self.point_feats_list[idx].device,
                        dtype=torch.int
                    )
                )
                self.game_pool[level].append(game)

            if len(self.game_pool[level]) > self.game_pool_size_limit:
                self.game_pool[level] = self.game_pool[level][-self.game_pool_size_limit:]

        random_level = random.choice(list(range(skill_level-2, skill_level+1)))
        self.first_state = torch.zeros(random_level, random_level, device=self.point_feats_list[0].device, dtype=torch.int)
        if len(self.game_pool[random_level]) > 0:
            return True, random_level
        else:
            return False, -1
        # self.gt_state = torch.zeros_like(self.cur_state)
        # self.gt_state[self.gt_edges[:,0], self.gt_edges[:,1]] = 1

    def init_state(self, game_level):
        for game in self.game_pool[game_level]:
            game['cur_state'] = game['first_state']

    def forward_gnn(self, point_preds, point_feats, state):

        B, N, _ = state.shape
        # edge_idxes = state.nonzero()
        # if len(edge_idxes) == 0:
        #     edge_idxes = edge_idxes[:, :2]
        # else:
        #     edge_idxes = edge_idxes[:, 1:] + edge_idxes[:,0:1] * N

        # batch_point_feats = point_feats.unsqueeze(0).repeat(B, 1, 1).view(B*N, -1)

        state_one_hot = F.one_hot(state.long(), num_classes=2).permute(0,3,1,2).float()
        # points_1 = self.point_preds_list[self.game_idx].view(1, N, 2).unsqueeze(1).repeat(B,N,1,1)
        # points_2 = self.point_preds_list[self.game_idx].view(1, N, 2).unsqueeze(2).repeat(B,1,N,1)
        points_1 = point_preds.view(B,1,N,2).repeat(1,N,1,1)
        points_2 = point_preds.view(B,N,1,2).repeat(1,1,N,1)
        points = torch.cat([points_1, points_2], dim=-1).permute(0,3,1,2)
        state_feats = torch.cat([state_one_hot, points], dim=1).view(B, 6, N*N)

        # pos_enc = self.pos_encoding(
        #     torch.zeros(B, N, N, self.channels_pos_enc, device=point_feats.device)
        # ).permute(0,3,1,2).view(B, self.channels_pos_enc, N*N)
        # state_feats = torch.cat([state_feats, pos_enc], dim=1)

        temp1 = point_feats.unsqueeze(1).repeat(1,N,1,1)
        temp2 = point_feats.unsqueeze(2).repeat(1,1,N,1)
        edge_feats = torch.cat([temp1, temp2], dim=-1)
        batch_edge_feats = edge_feats.permute(0,3,1,2).view(B,-1,N*N)


        if self.only_use_state:
            gnn_feats = state_feats
        else:
            gnn_feats = torch.cat([batch_edge_feats, state_feats], dim=1)

        scores, = self.gnn(gnn_feats.view(B, -1, N, N))
        scores = self.conv_desc(scores)
        scores = scores.view(B, N, N)

        # state_feats = torch.zeros(B*N, 2, device=point_feats.device, dtype=point_feats.dtype)
        # state_feats[edge_idxes[:,0], 0] = 1
        # state_feats[edge_idxes[:,1], 1] = 1

        # if not self.only_use_state:
        #     batch_edge_feats = torch.cat([batch_edge_feats, state_feats], dim=1)
        # else:
        #     batch_edge_feats = state_feats

        # gnn_feats, offsets = self.gnn(batch_point_feats.view(B, N, -1).permute(0,2,1))
        # scores = self.score_net(gnn_feats)


        # gnn_feats_1 = self.gnn(batch_point_feats, edge_idxes).view(B,N,-1).permute(0,2,1)
        # gnn_feats_2 = None
        # if self.apply_rev_gnn_feats:
        #     gnn_feats_2 = self.gnn(batch_point_feats, edge_idxes[[1,0]]).view(B,N,-1).permute(0,2,1)


        # scores = self.score_net(gnn_feats_1, state_feats, gnn_feats_2)

        return scores

    def sample_actions(self, scores):
        B, N, _ = scores.shape
        mask = scores > 0
        actions = []
        for i in range(B):
            if mask[i].sum() > 0:
                cur_action = random.choice(mask[i].nonzero())
            else:
                # choose the max one
                # cur_action = random.choice((mask[i] > -1).nonzero())
                cur_action = scores[i].argmax().item()
                cur_action = torch.tensor([cur_action // N, cur_action % N], device=scores.device)

            actions.append(cur_action)
        return actions



    """
    def state2polygons(self, edge_idxes):

        def merge_edges(poly):
            s = 0
            P = len(poly)
            while s < P:
                head = poly[s][-1]

                t = 0
                while t < P:
                    if s == t:
                        t += 1
                        continue
                    tail = poly[t][0]
                    if head == tail:
                        poly[s] = poly[s] + poly[t][1:]
                        del poly[t]
                        poly = merge_edges(poly)
                        P = len(poly)
                    t += 1
                s += 1
            return poly

        new_edges = merge_edges(edge_idxes)
        return new_edges
    """

    def state2polygons(self, edge_idxes):

        def merge_edges(poly):
            s = 0
            P = len(poly)
            while s < P:
                head_s = poly[s][0]
                tail_s = poly[s][-1]

                t = 0
                while t < P:
                    if s == t:
                        t += 1
                        continue
                    head_t = poly[t][0]
                    tail_t = poly[t][-1]
                    if head_t == tail_s:
                        poly[s] = poly[s] + poly[t][1:]
                        del poly[t]
                        poly = merge_edges(poly)
                        P = len(poly)
                    elif head_s == tail_t:
                        poly[t] = poly[t] + poly[s][1:]
                        del poly[s]
                        poly = merge_edges(poly)
                        P = len(poly)
                    t += 1
                s += 1
            return poly

        new_edges = merge_edges(edge_idxes)
        return new_edges



    def cal_iou(self, polygon1, polygon2):
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area
        iou = intersection_area / (union_area + 1e-8)
        return iou

    def get_first_state(self, cur_state, gt_edge):
        N = self.first_state.shape[0]

        if random.random() < self.gt_eps:
            idxes = torch.ones(N, device=gt_edge.device, dtype=torch.int)
            idxes[gt_edge[:,0]] = 0
            loop_edge = idxes.nonzero().view(-1)
            loop_edge = torch.stack([loop_edge, loop_edge], dim=1)
            gt_edge = torch.cat([gt_edge, loop_edge], dim=0)

            num_edges = random.randint(0, N-1)
            rand_perm = torch.randperm(N, device=gt_edge.device)
            rand_edges = gt_edge[rand_perm[:num_edges]]

            # left_edges = gt_edge[rand_perm[num_edges:]]

            cur_gt_state = self.first_state.clone()
            cur_gt_state[rand_edges[:,0], rand_edges[:,1]] = 1
            return cur_gt_state

        return self.first_state


        # gt_states.append(cur_gt_state)
        # cur_scores = torch.zeros(N, N, device=gt_edge.device) - 1
        # cur_scores[left_edges[:,0], left_edges[:,1]] = 1.
        # gt_scores.append(cur_scores)

        # gt_states = torch.stack(gt_states, dim=0)
        # gt_scores = torch.stack(gt_scores, dim=0)
        # pred_scores = self.forward_gnn(point_preds, point_feats, gt_states)



    def get_next_state(self, cur_state, action, points_ori, gt_polygons, gt_edge):
        reward = 0

        next_state = cur_state.clone()
        next_state[action[0], action[1]] += 1

        # check validity
        if (next_state.sum(dim=0) > 1).any() or (next_state.sum(dim=1) > 1).any():
            self.cnt_invalid_0 += 1
            return self.get_first_state(cur_state, gt_edge), -1, True, {}

        # polygons = permutations_to_polygons(next_state.unsqueeze(0).cpu(), points_ori.unsqueeze(0).cpu(), out='torch')
        edges_list = next_state.nonzero().tolist()


        poly_idxes_list = self.state2polygons(edges_list)
        # if next_state.nonzero().size(0) >= 7:
        #     pdb.set_trace()
        # poly_idxes_list = []

        valid_polygons = []
        return_polygons = []
        has_invalid = False
        for poly_idxes in poly_idxes_list:
            cur_polygon = points_ori[poly_idxes]
            # if len(poly_idxes) < 4:
            #     continue

            if len(poly_idxes) >= 4:
                cur_polygon_shape = Polygon(cur_polygon.tolist())
                if not cur_polygon_shape.is_valid:
                    reward += self.reward_invalid_2
                    self.cnt_invalid_2 += 1
                    has_invalid = True
                    break

            if poly_idxes[0] == poly_idxes[-1]:

                if len(poly_idxes) == 2:
                    # self looping points, ignore
                    continue

                if len(poly_idxes) >= 3 and len(poly_idxes) <= 4:
                    self.cnt_invalid_1 += 1
                    reward += self.reward_invalid_1
                    has_invalid = True
                    # break

                # cur_polygon_shape = Polygon(cur_polygon.tolist())

                if len(poly_idxes) >= 5:
                    if cur_polygon_shape.is_valid:
                        valid_polygons.append(cur_polygon_shape)
                        return_polygons.append(cur_polygon)

                # else:
                #     reward += self.reward_invalid_2
                #     self.cnt_invalid_2 += 1
                #     has_invalid = True
                    # break

        if has_invalid and self.early_stop:
            return self.get_first_state(cur_polygon, gt_edge), reward, True, {}

        if next_state.sum(dim=-1).all():
            # print(f"valid type: {self.cnt_valid}")
            # fully assigned, calculate rewards
            if len(valid_polygons) > 0:
                self.cnt_valid += 1
                # gt_polygons = self.gt_polygons_list[self.game_idx]
                # if gt_polygons is None:
                #     edges_list = self.gt_edges_list[self.game_idx].tolist()
                #     poly_idxes_list = self.state2polygons(edges_list)
                #     gt_polygons = [Polygon(points_ori[poly_idxes].tolist()) for poly_idxes in poly_idxes_list]

                iou_mat = np.zeros((len(valid_polygons), len(gt_polygons)))
                for i, pred_poly in enumerate(valid_polygons):
                    for j, gt_poly in enumerate(gt_polygons):
                        try:
                            iou_mat[i, j] = self.cal_iou(pred_poly, gt_poly)
                        except Exception:
                            print('invalid polygon found!')
                            iou_mat[i, j] = 0.1

                r, c = linear_sum_assignment(iou_mat, maximize=True)
                reward += iou_mat[r, c].sum() / len(gt_polygons)
                # if len(gt_polygons) > 1:
                #     pdb.set_trace()

                if len(valid_polygons) > len(gt_polygons):
                    reward -= (len(valid_polygons) - len(gt_polygons)) * self.extra_poly_penalty

                self.iou_scores.append(reward)

                states = {'pass_exam': reward > self.pass_reward_thre,
                          'return_polygons': return_polygons}
                # TODO: determine pass or not based on the iou of two polygon sets
                return self.get_first_state(cur_state, gt_edge), reward, True, states
            else:
                # no polygons, all points are self looping
                self.cnt_valid_2 += 1
                return self.get_first_state(cur_state, gt_edge), self.reward_valid2, True, {}

        return next_state, 0, False, {}


    def forward_train(self, game_level, exam_mode=False):

        num_limit_points = 512 * 512 * 8
        # min_start_size = 4 * 4 * 256
        batch_size = min(num_limit_points // (game_level**2), len(self.game_pool[game_level]))
        # if batch_size * game_level * game_level < min_start_size:
        #     return {}, {}


        games = random.choices(self.game_pool[game_level], k=batch_size)

        # scores = self.forward_gnn(self.point_feats_list[self.game_idx], self.cur_state.unsqueeze(0))
        states = torch.stack([game['cur_state'] for game in games], dim=0)
        point_feats = torch.stack([game['point_feats'] for game in games], dim=0)
        point_preds = torch.stack([game['point_preds'] for game in games], dim=0)
        point_preds_ori = torch.stack([game['point_preds_ori'] for game in games], dim=0)
        gt_edges = [game['gt_edges'] for game in games]

        scores = self.forward_gnn(point_preds, point_feats, states)
        actions = self.sample_actions(scores)
        actions = torch.stack(actions, dim=0)
        next_states, rewards, terminals, log_vars = [], [], [], []
        for i, game in enumerate(games):
            next_state, reward, terminal, log_var = self.get_next_state(
                states[i], actions[i], point_preds_ori[i], game['gt_polygons'],
                gt_edges[i]
            )
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(terminal)
            log_vars.append(log_var)

        next_states = torch.stack(next_states, dim=0)
        next_scores = self.forward_gnn(point_preds, point_feats, next_states)

        temp = torch.gather(scores, 1, actions[:, 0:1].unsqueeze(2).expand(-1,-1,game_level))
        Q_batch = torch.gather(temp, 2, actions[:, 1:2].unsqueeze(1))
        Q_batch = Q_batch.sum(dim=[1,2])

        Y_batch = torch.tensor([
            reward if terminal else \
            reward + self.gamma * next_pred.max().item() \
            for reward, terminal, next_pred in zip(rewards, terminals, next_scores)
        ], device=scores.device)

        loss_dqn = self.loss_fun(Q_batch, Y_batch)

        loss_row_col = self.cal_loss_row_col(states, scores)

        if self.add_gt_state and random.random() < self.gt_eps:
            loss_gt_state = self.cal_loss_gt_state(point_preds, point_feats, gt_edges, game_level)
        else:
            loss_gt_state = torch.zeros(1, device=scores.device)

        for i, game in enumerate(games):
            game['cur_state'] = next_states[i]

        # if self.cur_iter % 100 == 0:
        #     print(
        #         f"batch_size: {batch_size}, node_num: {game_level}, all: {self.cnt}, invalid0: {self.cnt_invalid_0}, invalid1: " +
        #         f"{self.cnt_invalid_1}, invalid2: {self.cnt_invalid_2}, valid: {self.cnt_valid}, valid2: {self.cnt_valid_2}, " +
        #         f"iou_scores: {0 if len(self.iou_scores) == 0 else sum(self.iou_scores)/len(self.iou_scores)}"
        #     )
        self.cur_iter += batch_size

        log_dict = {'pass_exam': []}
        for log_var in log_vars:
            log_dict['pass_exam'].append(log_var.get('pass_exam', False))
        log_dict['batch_size'] = batch_size
        log_dict['iter'] = self.cur_iter
        log_dict['invalid0'] = self.cnt_invalid_0
        log_dict['invalid1'] = self.cnt_invalid_1
        log_dict['invalid2'] = self.cnt_invalid_2
        log_dict['valid'] = self.cnt_valid
        log_dict['valid2'] = self.cnt_valid_2
        log_dict['game_level'] = game_level
        log_dict['iou_scores'] = 0 if len(self.iou_scores) == 0 else sum(self.iou_scores)/len(self.iou_scores)


        return {'loss_dqn': loss_dqn * self.loss_dqn_weight,
                'loss_row_col': loss_row_col * self.loss_row_col_weight,
                'loss_gt_state': loss_gt_state * self.loss_gt_state_weight}, log_dict

    def cal_loss_gt_state(self, point_preds, point_feats, gt_edges, N):
        gt_states = []
        gt_scores = []

        for gt_edge in gt_edges:
            idxes = torch.ones(N, device=gt_edge.device, dtype=torch.int)
            idxes[gt_edge[:,0]] = 0
            loop_edge = idxes.nonzero().view(-1)
            loop_edge = torch.stack([loop_edge, loop_edge], dim=1)
            gt_edge = torch.cat([gt_edge, loop_edge], dim=0)

            num_edges = random.randint(0, N-1)
            rand_perm = torch.randperm(N, device=gt_edge.device)
            rand_edges = gt_edge[rand_perm[:num_edges]]
            left_edges = gt_edge[rand_perm[num_edges:]]

            cur_gt_state = self.first_state.clone()
            cur_gt_state[rand_edges[:,0], rand_edges[:,1]] = 1
            gt_states.append(cur_gt_state)
            cur_scores = torch.zeros(N, N, device=gt_edge.device) - 1
            cur_scores[left_edges[:,0], left_edges[:,1]] = 1.
            gt_scores.append(cur_scores)

        gt_states = torch.stack(gt_states, dim=0)
        gt_scores = torch.stack(gt_scores, dim=0)
        pred_scores = self.forward_gnn(point_preds, point_feats, gt_states)

        mask = gt_scores > 0
        if mask.sum().item() > 0:
            loss = self.loss_fun(pred_scores[mask], gt_scores[mask])
        else:
            loss = torch.zeros(1, device=point_preds.device)

        return loss



    def cal_loss_row_col(self, states, scores):
        targets = []
        for i, state in enumerate(states):
            edge_idxes = state.nonzero()
            rows = edge_idxes[:, 0]
            cols = edge_idxes[:, 1]
            target = torch.ones_like(state, dtype=torch.float)
            target[rows, :] = self.reward_invalid_0
            target[:, cols] = self.reward_invalid_0
            targets.append(target)
        targets = torch.stack(targets, dim=0)
        mask = targets < 0
        if mask.sum().item() > 0:
            loss_row_col = self.loss_fun(scores[mask], targets[mask])
        else:
            loss_row_col = torch.zeros(1, device=scores.device)

        if (~mask).sum().item() > 0:
            mask2 = scores[~mask] < self.reward_invalid_1
            if mask2.sum().item() > 0:
                loss_row_col_2 = self.loss_fun(
                    scores[~mask][mask2],
                    torch.zeros(mask2.sum(), device=scores.device).fill_(-0.5)
                )
            else:
                loss_row_col_2 = torch.zeros(1, device=scores.device)

        return loss_row_col + loss_row_col_2


    def forward_test(self):

        B, N, C = point_feats.shape
        if B == 0 or N == 0:
            return False

        # edge_idxes = []
        node_feats = []
        node_labels = []
        node_masks = []
        batch_start_idxes = []
        group_idxes = []
        start_idx = 0
        gt_permute_1 = graph_targets.argmax(dim=-1)
        cur_groud_idx = 0

        for i in range(B):
            num_cur_nodes = comp_mask[i].any(dim=-1).sum().item()
            edge_idx = comp_mask[i].nonzero()
            # edge_idxes.append(edge_idx + start_idx)
            node_label = gt_permute_1[i]
            node_mask = (graph_targets[i] > 0).any(dim=-1)

            node_feats.append(point_feats[i, :num_cur_nodes])
            node_labels.append(node_label[:num_cur_nodes])
            node_masks.append(node_mask[:num_cur_nodes])
            batch_start_idxes.append(start_idx)

            start_idx += num_cur_nodes

        batch_start_idxes.append(start_idx)

        node_feats = torch.cat(node_feats, dim=0)
        node_labels = torch.cat(node_labels, dim=0)
        # edge_idxes = torch.cat(edge_idxes, dim=0).permute(1,0)
        node_masks = torch.cat(node_masks, dim=0)

        self.cur_state = dict(
            idxes=torch.zeros(node_feats.shape[0], dtype=torch.int) - 1,
            node_feats=node_feats,
            node_labels=node_labels,
            action=None, # init_state
            reward=None
        )
        self.replay_memory = []

        return True

    def test(self):
        if B == 0 or N == 0:
            return {'deep_q_loss': torch.zeros(1, device=point_feats.device)}, {}

        gnn_feat, offset = self.gnn(point_feats.permute(0,2,1),
                                    point_preds.permute(0,2,1)) # B, C, N
        offset_preds = point_preds + offset * self.correction_radius
        gnn_cls_labels = (graph_targets > 0).any(dim=-1).to(torch.uint8)
        node_valid_mask = comp_mask.any(dim=-1)
        gnn_cls_preds = self.fc_gnn_cls(gnn_feat.permute(0,2,1)) # B, N, 2

        return losses, state

    def forward_test_dqn(self, point_feats, point_preds, point_preds_ori):
        N, _ = point_preds.shape
        cur_state = torch.zeros(N, N, dtype=torch.int, device=point_preds.device)
        polygons = []

        for i in range(N):
            scores = self.forward_gnn(
                point_preds.unsqueeze(0),
                point_feats.unsqueeze(0),
                cur_state.unsqueeze(0)
            )
            cur_action = scores[0].argmax().item()
            cur_action = torch.tensor([cur_action // N, cur_action % N], device=scores.device)
            cur_state[cur_action[0], cur_action[1]] = 1
        polygons = self.state2polygons(cur_state.nonzero().tolist())
        return polygons


    def forward_test_global(self, point_feats, point_preds, point_preds_ori, comp_mask,
                            batch_size_list, **kwargs):
        # if self.remove_last:
        #     point_feats = point_feats[:, :, :-2]
        # point_angles = self.fc_angle(point_feats)
        # B, N, num_angle = point_angles.shape

        for batch_idx in range(len(batch_size_list)):
            start_idx = 0
            for size in batch_size_list[batch_idx]:
                comp_point_feats = point_feats[batch_idx][start_idx:start_idx+size]
                comp_point_preds = point_preds[batch_idx][start_idx:start_idx+size]
                comp_point_preds_ori = point_preds_ori[batch_idx][start_idx:start_idx+size]
                polygons = self.forward_test_dqn(comp_point_feats, comp_point_preds, comp_point_preds_ori)

        gnn_feat, offset = self.gnn(point_feats.permute(0,2,1), point_preds.permute(0,2,1)) # B, C, N
        # scores_1 = F.softmax(self.scorenet1(gnn_feat), dim=-1)
        # scores_2 = F.softmax(self.scorenet2(gnn_feat), dim=-1)
        scores_1 = self.scorenet1(gnn_feat)
        scores_2 = self.scorenet2(gnn_feat)
        scores_1 = torch.where(comp_mask==1, scores_1, -1e8)
        scores_2 = torch.where(comp_mask==1, scores_2, -1e8)
        scores_1 = F.softmax(scores_1, dim=-1)
        scores_2 = F.softmax(scores_2, dim=-1)

        if self.single_permute:
            scores = scores_1
        else:
            scores = scores_1 + scores_2.transpose(1, 2)

        # scores = scores_to_permutations(scores, 0.1)
        scores = scores_to_permutations(scores, ignore_thre=0.0)

        # temp = self.normalize_coordinates(point_preds, img.shape[-1], 'normalized').detach()
        polygon = permutations_to_polygons(scores, point_preds_ori, out='numpy')

        return polygon, [point_pred_ori for point_pred_ori in point_preds_ori.detach().cpu()]

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test_global(**kwargs)


