import torch

import torch.nn as nn
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmcv.runner import BaseModule
import torch.nn.functional as F
import pdb
import torch_geometric.nn.models as geom_models
from torch_geometric.nn import conv as geom_conv
import random
# from utils import scores_to_permutations, permutations_to_polygons

from ..builder import HEADS
from rsidet.models.utils import build_linear_layer
from shapely.geometry import Polygon
from positional_encodings.torch_encodings import PositionalEncoding2D
from .. import builder
import warnings

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
class DeepQNetV9(BaseModule):

    def __init__(
        self, node_net=None, in_channels=512, hidden_channels=64, num_gnn_layers=3, return_polygons=False,
        max_iter=2000, final_eps=1e-4, init_eps=0.1, extra_poly_penalty=0.2,
        num_limit_points_per_batch=512*512*8, replay_memory_size=32, gamma=0.99,
        num_gnn_hop=10, apply_rev_gnn_feats=True, only_use_state=False, pass_reward_thre=0.8,
        gt_eps=0.5, channels_pos_enc=2, out_gnn_channels=256, add_gt_state=False,
        loss_row_col_weight=100., loss_dqn_weight=1.0,
        loss_gt_state_pos_weight=10., loss_gt_state_neg_weight=1.,
        reward_invalid_0=-1., reward_invalid_1=-0.5, reward_invalid_2=-0.2, reward_valid2=-0.5,
        rand_sample_action=False, early_stop=True, game_pool_size_limit = 256,
        pos_terminal_weight=100., terminal_weight=10., action_sample_type='random_positive',
        num_max_points=32, max_batch_size=256, max_base_size=80
    ):
        super(DeepQNetV9, self).__init__()

        self.act_net = builder.build_backbone(node_net)
        self.target_net = builder.build_backbone(node_net)

        # self.act_net = AttentionalGNN(
        #     hidden_channels, num_gnn_layers, in_channels=in_channels, out_channels=1
        # )

        # self.target_net = AttentionalGNN(
        #     hidden_channels, num_gnn_layers, in_channels=in_channels, out_channels=1
        # )
        # self.terminate_node = nn.Parameter(torch.randn(in_channels, requires_grad=True))
        # self.act_net.terminate_node = self.terminate_node
        self.terminate_node = nn.Parameter(torch.zeros(in_channels) - 1)

        self.cse_loss_fun = nn.CrossEntropyLoss()
        self.loss_fun = nn.SmoothL1Loss(reduction='none')
        # self.loss_fun = nn.SmoothL1Loss()

        # self.gnn = geom_models.GAT(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     num_layers=num_gnn_layers,
        #     K=num_gnn_hop,
        # )
        # self.gnn = geom_conv.TAGConv(in_channels=in_channels, out_channels=hidden_channels, K=num_gnn_hop)

        # self.score_net = ScoreNet(hidden_channels * 2)
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
        self.loss_gt_state_pos_weight = loss_gt_state_pos_weight
        self.loss_gt_state_neg_weight = loss_gt_state_neg_weight
        self.pos_terminal_weight = pos_terminal_weight
        self.terminal_weight = terminal_weight
        self.reward_invalid_0 = reward_invalid_0
        self.reward_invalid_1 = reward_invalid_1
        self.reward_invalid_2 = reward_invalid_2
        self.reward_valid2 = reward_valid2
        self.rand_sample_action = rand_sample_action
        self.early_stop = early_stop
        self.game_pool = {}
        self.game_pool_size_limit = game_pool_size_limit
        self.action_sample_type = action_sample_type
        self.num_max_points = num_max_points
        self.max_batch_size = max_batch_size
        self.max_base_size = max_base_size

    def greedy_arange(self, sizes, max_base_size):

        sorted_idxes = np.argsort(np.array(sizes))[::-1]
        start_idx = (sizes > max_base_size).sum()
        if (sizes > max_base_size).all():
            return [], []
        base_size = sizes[sorted_idxes[start_idx]]
        batch_idx_list = [[sorted_idxes[start_idx]]]
        batch_size_list = [[base_size]]
        cur_bin = []
        cur_sizes = []
        cur_size = 0
        for i, idx in enumerate(sorted_idxes[start_idx+1:]):
            if sizes[idx] < 4:
                break
            if sizes[idx] > max_base_size:
                continue
            if cur_size + sizes[idx] <= base_size:
                cur_bin.append(idx)
                cur_sizes.append(sizes[idx])
                cur_size += sizes[idx]
            else:
                batch_idx_list.append(cur_bin)
                batch_size_list.append(cur_sizes)
                cur_bin = [idx]
                cur_size = sizes[idx]
                cur_sizes = [sizes[idx]]

        if len(cur_bin) > 0:
            batch_idx_list.append(cur_bin)
            batch_size_list.append(cur_sizes)

        return batch_idx_list, batch_size_list

    def set_environment(self, point_feats_list, gt_edges_list, point_preds_list,
                        point_preds_ori_list, cur_iter=0):

        self.point_feats_list = [x.detach() for x in point_feats_list]
        # self.point_feats_list = [torch.cat([x1.detach(), x2], dim=1) for x1, x2 in zip(point_feats_list, point_preds_list)]
        self.gt_edges_list = gt_edges_list
        self.point_preds_list = [x.detach() for x in point_preds_list]
        self.point_preds_ori_list = point_preds_ori_list
        self.gt_polygons_list = [None] * len(self.gt_edges_list)
        self.cur_iter = cur_iter
        self.poly_idxes_list = [self.state2polygons(edges.tolist()) for edges in self.gt_edges_list]

    def sample_games(self):


        ## TODO: improve the greedy_arange algorithm
        sizes = np.array([len(x) for x in self.point_feats_list])
        batch_idx_list, batch_size_list = self.greedy_arange(sizes, self.max_base_size)
        if len(batch_idx_list) == 0:
            return None

        num_limit_points = 512 * 512 * 32
        # min_start_size = 4 * 4 * 128
        base_size = sizes[batch_idx_list[0][0]]
        num_max_batch = num_limit_points // base_size ** 2
        assert base_size <= self.max_base_size

        ## TODO: shuffle the list
        if len(batch_idx_list) > num_max_batch:
            batch_idx_list = batch_idx_list[:num_max_batch]
            batch_size_list = batch_size_list[:num_max_batch]


        device = self.point_feats_list[0].device
        # batch_size = min(num_limit_points // (game_level**2), len(self.game_pool[game_level]))
        # batch_size = min(batch_size, self.max_batch_size)

        new_batch_idx_list = []
        new_batch_size_list = []
        batch_gt_polygons = []
        new_base_size = 0
        for i, batch_idxes in enumerate(batch_idx_list):
            num_gt_nodes = 0
            gt_polygons = []
            for idx in batch_idxes:
                gt_poly_sizes = [len(x) for x in self.poly_idxes_list[idx]]
                num_gt_nodes += sum(gt_poly_sizes) + len(gt_poly_sizes)
                gt_polygons.append(
                    [Polygon(self.point_preds_ori_list[idx][poly_idxes].tolist()) for poly_idxes in self.poly_idxes_list[idx]]
                )

            if num_gt_nodes + 1 >= self.num_max_points:
                continue

            batch_gt_polygons.append(gt_polygons)
            new_batch_idx_list.append(batch_idxes)
            new_batch_size_list.append(batch_size_list[i])
            new_base_size = max(new_base_size, sum(batch_size_list[i]) + len(batch_idxes))

        base_size = new_base_size

        batch_idx_list = new_batch_idx_list
        batch_size_list = new_batch_size_list

        if len(batch_idx_list) == 0:
            return None

        batch_points = torch.zeros(
            len(batch_idx_list),
            base_size, 2, device=device
        )
        batch_points_ori = torch.zeros(
            len(batch_idx_list),
            base_size, 2, device=device
        )
        batch_point_feats = torch.zeros(
            len(batch_idx_list),
            base_size, self.point_feats_list[0].size(1),
            device=device
        )
        dummy_points = torch.zeros(1, 2, device=device) - 1
        dummy_feats = torch.zeros(1, self.point_feats_list[0].size(1), device=device) - 1

        for i, batch_idxes in enumerate(batch_idx_list):
            cur_batch_points = torch.cat([torch.cat([self.point_preds_list[idx], dummy_points]) for idx in batch_idxes])
            cur_batch_points_ori = torch.cat([torch.cat([self.point_preds_ori_list[idx], dummy_points]) for idx in batch_idxes])
            cur_batch_point_feats = torch.cat([torch.cat([self.point_feats_list[idx], dummy_feats]) for idx in batch_idxes])

            assert len(cur_batch_points) <= base_size
            batch_points[i, :len(cur_batch_points)] = cur_batch_points
            batch_points_ori[i, :len(cur_batch_points_ori)] = cur_batch_points_ori
            batch_point_feats[i, :len(cur_batch_point_feats)] = cur_batch_point_feats

        games = dict(
            points=batch_points,
            points_ori=batch_points_ori,
            point_feats=batch_point_feats,
            batch_sizes=batch_size_list,
            batch_idxes=batch_idx_list,
            poly_idxes_list=self.poly_idxes_list,
            batch_gt_polygons=batch_gt_polygons
        )

        return games

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

        for level in range(skill_level, skill_level+1):
            if not level in self.game_pool.keys():
                self.game_pool[level] = []

            valid_idx = (sizes == level).nonzero()[0]
            for idx in valid_idx:
                point_preds_ori = self.point_preds_ori_list[idx]
                edges_list = self.gt_edges_list[idx].tolist()
                poly_idxes_list = self.state2polygons(edges_list)
                poly_idxes_list = [poly_idxes for poly_idxes in poly_idxes_list if len(poly_idxes) >= 5]
                gt_polygons = [Polygon(point_preds_ori[poly_idxes].tolist()) for poly_idxes in poly_idxes_list]
                valid = True
                for gt_poly in gt_polygons:
                    if not gt_poly.is_valid:
                        valid = False

                if not valid:
                    continue

                game = dict(
                    point_feats = self.point_feats_list[idx],
                    gt_edges = self.gt_edges_list[idx],
                    point_preds = self.point_preds_list[idx],
                    point_preds_ori = self.point_preds_ori_list[idx],
                    gt_polygons = gt_polygons,
                    poly_idxes_list=poly_idxes_list,
                    first_state = torch.zeros(
                        level, level,
                        device=self.point_feats_list[idx].device,
                        dtype=torch.int
                    ),
                    cur_state = torch.zeros(
                        level, level,
                        device=self.point_feats_list[idx].device,
                        dtype=torch.int
                    ),
                    point_seqs = torch.zeros(
                        self.num_max_points,
                        device=self.point_feats_list[idx].device,
                        dtype=torch.long
                    ) - 1
                )
                self.game_pool[level].append(game)

            if len(self.game_pool[level]) > self.game_pool_size_limit:
                self.game_pool[level] = self.game_pool[level][-self.game_pool_size_limit:]

        random_level = random.choice(list(range(skill_level, skill_level+1)))
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

    def update_target_net(self):
        self.target_net.load_state_dict(self.act_net.state_dict())

    def update_act_net(self):
        self.act_net.load_state_dict(self.target_net.state_dict())

    def forward_gnn(self, point_preds, point_feats, point_seqs, batch_sizes, type='act'):

        B, N, _ = point_preds.shape
        T = self.num_max_points
        # if point_seqs.max() >= N:
        #     pdb.set_trace()

        point_seq_feats = torch.zeros(B, T, N, device=point_preds.device, dtype=point_preds.dtype)
        attn_mask = ~torch.eye(N, N, device=point_preds.device, dtype=torch.bool)
        attn_mask = attn_mask.view(1, N, N).repeat(B, 1, 1)

        for idx in range(B):
            start_idx = 0
            for idx2, size in enumerate(batch_sizes[idx]):
                seq_idxes = (point_seqs[idx][idx2] >= 0).nonzero().view(-1)
                point_seq_feats[idx, seq_idxes, point_seqs[idx][idx2][seq_idxes] + start_idx] = 1
                attn_mask[idx, start_idx:start_idx+size+1, start_idx:start_idx+size+1] = 0
                # attn_mask[idx, start_idx:start_idx+size, N] = 0
                # attn_mask[idx, N, start_idx:start_idx+size] = 0
                # attn_mask[idx, N, N] = 0
                start_idx += size + 1

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.act_net.num_heads, 1, 1).view(-1, N, N)

        # seq_idxes = (point_seqs >= 0).nonzero()
        # if not (len(seq_idxes) == 0 or seq_idxes[:,1].max() < T):
        #     pdb.set_trace()
        # point_seq_feats[seq_idxes[:,0], seq_idxes[:,1], point_seqs[seq_idxes[:,0], seq_idxes[:,1]]] = 1

        if self.only_use_state:
            node_feats = torch.cat([point_preds.permute(0,2,1), point_seq_feats], dim=1)
        else:
            node_feats = torch.cat([point_preds.permute(0,2,1), point_feats.permute(0,2,1), point_seq_feats], dim=1)

        # terminate_node = self.terminate_node.view(1, -1, 1).repeat(B, 1, 1)
        # node_feats = torch.cat([node_feats, terminate_node], dim=2)

        if type == 'act':
            scores = self.act_net(node_feats, attn_mask=attn_mask)
        else:
            scores = self.target_net(node_feats, attn_mask=attn_mask)

        scores = scores.view(B, N)
        return scores

    def sample_actions(self, scores, point_seqs, batch_sizes, method='random_positive', random_eps=0.1):

        if method == 'node_based':
            B, N = scores.shape

            batch_actions = []
            for i in range(B):
                start_idx = 0
                actions = []
                for j, seqs in enumerate(point_seqs[i]):
                    if (seqs == batch_sizes[i][j]).any():
                        # terminated already
                        actions.append(batch_sizes[i][j])
                        continue

                    T = seqs.size(0)
                    seq_idxes = (seqs >= 0).nonzero().view(-1)
                    temp = torch.bincount(seqs[seqs>=0])
                    degrees = torch.zeros(T, device=seqs.device, dtype=torch.long)
                    degrees[:len(temp)] = temp
                    degrees = torch.where(seqs >= 0, degrees[seqs], 0)

                    arange = torch.arange(T, device=seqs.device, dtype=torch.long)
                    cur_poly_start_idx = torch.where(degrees == 2, arange + 1, 0).max()
                    cur_poly_end_idx = (seqs >= 0).sum()

                    if cur_poly_end_idx >= T-2:
                        action = batch_sizes[i][j]
                    else:
                        valid_mask = torch.ones(batch_sizes[i][j]+1, device=seqs.device, dtype=torch.bool)
                        invalid_seq_idxes = seqs[:cur_poly_end_idx]
                        valid_seq_idxes = seqs[cur_poly_start_idx] if seqs[cur_poly_start_idx] >= 0 else None

                        valid_mask[invalid_seq_idxes] = 0
                        if valid_seq_idxes is not None and cur_poly_start_idx + 4 <= cur_poly_end_idx:
                            valid_mask[valid_seq_idxes] = 1

                        assert valid_mask.any()
                        cur_scores = scores[i, valid_mask.nonzero().view(-1) + start_idx]
                        action = cur_scores.argmax()
                        action = valid_mask.nonzero().view(-1)[action]
                        # if (seqs == action).any():
                        #     pdb.set_trace()

                    actions.append(action.item())
                    start_idx += batch_sizes[i][j] + 1

                batch_actions.append(actions)

            return batch_actions


            T = point_seqs.size(1)
            seq_idxes = (point_seqs >= 0).nonzero()
            assert len(seq_idxes) == 0 or point_seqs.max() < N - 1

            point_seq_feats = torch.zeros(B, T, N, device=point_seqs.device, dtype=torch.int)
            point_seq_feats[seq_idxes[:,0], seq_idxes[:,1], point_seqs[seq_idxes[:,0], seq_idxes[:,1]]] = 1
            degrees = point_seq_feats.sum(dim=1)

            point_seq_degrees = torch.zeros_like(point_seqs, dtype=torch.long)
            point_seq_degrees[seq_idxes[:,0], seq_idxes[:,1]] = \
                    degrees[seq_idxes[:, 0], point_seqs[seq_idxes[:,0], seq_idxes[:,1]]]

            arange = torch.arange(T, device=point_seqs.device, dtype=torch.long).view(1, T).repeat(B, 1)
            cur_poly_start_idx = torch.where(point_seq_degrees == 2, arange + 1, 0).max(dim=1)[0]
            cur_poly_end_idx = (point_seqs >= 0).sum(dim=1)

            valid_mask = torch.ones_like(scores, dtype=torch.bool)
            invalid_seq_idx_mask = seq_idxes[:,1] < cur_poly_end_idx[seq_idxes[:,0]]
            valid_seq_idx_mask = seq_idxes[:,1] == cur_poly_start_idx[seq_idxes[:,0]]
            valid_batch_mask = cur_poly_start_idx + 4 <= cur_poly_end_idx
            temp_mask = valid_batch_mask[seq_idxes[valid_seq_idx_mask, 0]]


            valid_point_idxes = point_seqs[seq_idxes[valid_seq_idx_mask, 0][temp_mask],
                                           seq_idxes[valid_seq_idx_mask, 1][temp_mask]]

            invalid_point_idxes = point_seqs[seq_idxes[invalid_seq_idx_mask, 0], seq_idxes[invalid_seq_idx_mask, 1]]

            valid_mask[seq_idxes[invalid_seq_idx_mask, 0], invalid_point_idxes] = 0
            valid_mask[seq_idxes[valid_seq_idx_mask, 0][temp_mask], valid_point_idxes] = 1
            start_mask = cur_poly_end_idx == 0
            valid_mask[start_mask, N-1] = 0

            masked_scores = torch.where(valid_mask, scores, -1e8)
            actions = masked_scores.argmax(dim=1)
            pos_mask = (masked_scores > 0).any(dim=1)
            # if random.random() < self.init_eps:
            #     for idx in pos_mask.nonzero().view(-1):
            #         actions[idx] = random.choice((masked_scores > 0)[idx].nonzero().view(-1))

            # if (actions == 0).sum() >= 5:
            #     pdb.set_trace()

        return actions



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



    def cal_iou(self, polygon1, polygon2):
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area
        iou = intersection_area / (union_area + 1e-8)
        return iou

    def get_first_state(self, point_seqs, gt_poly_idxes_list):

        return torch.zeros_like(point_seqs) - 1
        N = self.first_state.shape[0]

        if random.random() < self.gt_eps:
            pdb.set_trace()
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
            cur_gt_state[rand_edges[:,1], rand_edges[:,0]] = 1
            return cur_gt_state
        else:
            return torch.zeros_like(point_seqs) - 1

        # gt_states.append(cur_gt_state)
        # cur_scores = torch.zeros(N, N, device=gt_edge.device) - 1
        # cur_scores[left_edges[:,0], left_edges[:,1]] = 1.
        # gt_scores.append(cur_scores)

        # gt_states = torch.stack(gt_states, dim=0)
        # gt_scores = torch.stack(gt_scores, dim=0)

    def dfs(self, graph, visited, node, component):
        visited[node] = True
        component.append(node)

        neighbors = torch.nonzero(graph[node]).view(-1).tolist()
        for neighbor in neighbors:
            if not visited[neighbor]:
                self.dfs(graph, visited, neighbor, component)

    def get_connected_components(self, graph):
        n = graph.size(0)
        visited = torch.zeros(n, dtype=torch.bool)
        components = []

        for node in range(n):
            if not visited[node]:
                component = []
                self.dfs(graph, visited, node, component)
                components.append(component)

        return components

    def cal_rewards(self, pred_polygons_list, pred_polygons, gt_polygons=None):

        reward = 0
        valid_polygons = []
        valid_polygons_list = []
        has_invalid = False
        for i, polygon in enumerate(pred_polygons):
            if len(pred_polygons_list[i]) >= 4:
                # print(components)
                # cur_polygon = points_ori[batch_idx][component]
                # cur_polygon_shape = Polygon(cur_polygon.tolist())
                if not polygon.is_valid:
                    reward = self.reward_invalid_2
                    # self.cnt_invalid_2 += 1
                    has_invalid = True
                    break
                else:
                    valid_polygons.append(polygon)
                    valid_polygons_list.append(pred_polygons_list[i])

            elif len(pred_polygons_list[i]) > 1 and len(pred_polygons_list[i]) <= 3:
                reward = self.reward_invalid_1
                # self.cnt_invalid_1 += 1
                has_invalid = True
                pdb.set_trace()
                break

        if has_invalid or gt_polygons is None:
            return reward, valid_polygons_list

        if len(valid_polygons) > 0:
            iou_mat = np.zeros((len(valid_polygons), len(gt_polygons)))
            for i, pred_poly in enumerate(valid_polygons):
                for j, gt_poly in enumerate(gt_polygons):
                    try:
                        with warnings.catch_warnings(record=True) as w:
                            iou_mat[i, j] = self.cal_iou(pred_poly, gt_poly)
                            if w:
                                pass
                    except Exception:
                        print('invalid gt polygon found!')
                        iou_mat[i, j] = 0.1

            r, c = linear_sum_assignment(iou_mat, maximize=True)
            reward = iou_mat[r, c].sum() / len(gt_polygons) * 1

            if len(valid_polygons) > len(gt_polygons):
                reward -= (len(valid_polygons) - len(gt_polygons)) * self.extra_poly_penalty

        return reward, valid_polygons_list


    def get_next_states(self, point_seqs, actions, points_ori, gt_polygons, batch_sizes):

        B, N, _ = points_ori.shape
        T = self.num_max_points

        assert len(point_seqs) == B
        assert len(actions) == B
        if gt_polygons is not None:
            assert len(gt_polygons) == B
        # assert len(gt_poly_idxes_list) == B

        # batch_terminates = torch.zeros(B, device=points_ori.device, dtype=torch.bool)
        batch_terminals = []
        batch_next_seqs = []
        batch_rewards = []
        ious = {}
        return_polygons = {}

        for i in range(B):
            terminals = []
            next_seqs = []
            rewards = []
            assert len(actions[i]) == len(point_seqs[i])
            offset = 0
            for j in range(len(actions[i])):
                action = actions[i][j]
                seqs = point_seqs[i][j]
                N = batch_sizes[i][j]
                terminals.append(action >= N)
                cur_poly_end_idx = (seqs >= 0).sum()
                temp = seqs.clone()
                # if action >= N:
                temp[cur_poly_end_idx] = action
                next_seqs.append(temp)

                reward = 0
                if action >= N and ~(seqs == N).any():
                    components = []
                    pred_poly_idxes = (seqs[seqs >= 0] + offset).tolist()
                    start_idx = 0
                    for idx, poly_idx in enumerate(pred_poly_idxes):
                        if idx > start_idx and poly_idx == pred_poly_idxes[start_idx]:
                            components.append(pred_poly_idxes[start_idx:idx])
                            start_idx = idx + 1

                    # if len(components) > 0:
                    #     pdb.set_trace()
                    pred_polygons = [Polygon(points_ori[i, component].tolist()) for component in components]
                    pred_polygons_list = [points_ori[i, component].tolist() for component in components]
                    reward, valid_polygons = self.cal_rewards(
                        pred_polygons_list, pred_polygons,
                        gt_polygons[i][j] if gt_polygons is not None else None
                    )

                    if not N in ious:
                        ious[N] = []
                        return_polygons[N] = []
                    ious[N].append(reward)
                    return_polygons[N].extend(valid_polygons)


                rewards.append(reward)
                offset += N + 1


            batch_next_seqs.append(next_seqs)
            # batch_terminates[i] = torch.tensor(terminates, device=points_ori.device).all()
            batch_terminals.append(terminals)
            batch_rewards.append(rewards)

        return batch_next_seqs, batch_rewards, batch_terminals, (ious, return_polygons)


    def evaluate(self, games, net_type='act'):

        points = games['points']
        points_ori = games['points_ori']
        point_feats = games['point_feats']
        batch_sizes = games['batch_sizes']
        batch_idxes = games['batch_idxes']
        batch_gt_polygons = games.get('batch_gt_polygons', None)

        B, N, _ = points.shape


        active_games = torch.ones(B, device=points.device, dtype=torch.bool)
        cur_point_seqs = []
        success_vector = np.zeros(B, dtype=np.int)

        for i in range(B):
            point_seqs = []
            sizes = []
            for j, size in enumerate(batch_sizes[i]):
                point_seqs.append(torch.zeros(self.num_max_points, dtype=torch.long,
                                              device=points.device)-1)
                sizes.append(size)
            cur_point_seqs.append(point_seqs)

        ious = {}
        return_polygons = {}
        for i in range(N * 2):
            if active_games.sum() == 0:
                break
            game_idx_list = active_games.nonzero().view(-1).tolist()
            # idx_point_seqs = [cur_point_seqs[x] for x in game_idx_list]
            cur_batch_sizes = [batch_sizes[x] for x in game_idx_list]

            scores = torch.zeros(B, N, device=points.device)
            with torch.no_grad():
                active_scores = self.forward_gnn(
                    points[active_games],
                    point_feats[active_games],
                    [cur_point_seqs[x] for x in game_idx_list],
                    cur_batch_sizes,
                    type=net_type
                )
                scores[game_idx_list] = active_scores

            active_actions = self.sample_actions(
                scores,
                cur_point_seqs,
                batch_sizes,
                method=self.action_sample_type
            )

            # actions = torch.stack(actions, dim=0)
            # next_states, rewards, terminals, log_vars = [], [], [], []
            # actions = torch.zeros(B, dtype=torch.long, device=scores.device) - 1
            # actions[active_games] = active_actions

            next_point_seqs, rewards, batch_terminals, (cur_ious, cur_return_polygons) = self.get_next_states(
                cur_point_seqs,
                active_actions,
                points_ori,
                batch_gt_polygons,
                batch_sizes
                # points_ori[active_games],
                # [batch_gt_polygons[x] for x in game_idx_list],
                # [gt_poly_idxes_list[x] for x in game_idx_list],
                # cur_batch_sizes
            )
            terminals = [torch.tensor(x, device=points.device, dtype=torch.bool).all() for x in batch_terminals]
            terminals = torch.stack(terminals)
            for key, value in cur_ious.items():
                if not key in ious:
                    ious[key] = []
                    return_polygons[key] = []
                ious[key] += value
                return_polygons[key] += cur_return_polygons[key]

            # success_vector = success_vector | np.array([x.get('pass_exam', False) for x in log_vars])

            active_games = active_games & (~terminals)
            # active_games[active_games == 1] = ~terminals
            cur_point_seqs = next_point_seqs

        success_rate = {}
        success_iou = {}
        success_cnt = {}
        all_cnt = {}
        for key, value in ious.items():
            value_np = np.array(value)
            success_rate[key] = (value_np >= self.pass_reward_thre).mean()
            all_cnt[key] = len(value_np)
            success_cnt[key] = (value_np > 0).sum()
            success_iou[key] = value_np[value_np > 0].mean() if success_cnt[key] > 0 else 0

        return dict(
            success_rate=success_rate,
            success_iou=success_iou,
            success_cnt=success_cnt,
            all_cnt=all_cnt,
            return_polygons=return_polygons
        )

    def init_state(self, game_level):
        for game in self.game_pool[game_level]:
            game['cur_state'] = game['first_state']


    def forward_train(self, games):

        batch_size = len(games['batch_sizes'])
        losses = {}
        log_vars = [{} for _ in range(batch_size)]
        if self.loss_dqn_weight > 0:
            loss_dqn, log_vars = self.cal_loss_dqn(games)
        else:
            loss_dqn = dict(
                loss_dqn = torch.zeros(1, device=games['points'][0].device),
                loss_terminal = torch.zeros(1, device=games['points'][0].device),
            )
        losses.update(loss_dqn)

        if self.add_gt_state:
            loss_gt_state = self.cal_loss_gt_state(games)
        else:
            loss_gt_state = dict(
                loss_gt_state_pos=torch.zeros(1, device=games['points'][0].device),
                loss_gt_state_neg=torch.zeros(1, device=games['points'][0].device),
            )

        losses.update(loss_gt_state)

        self.cur_iter += batch_size

        log_dict = {'pass_exam': []}
        for log_var in log_vars:
            log_dict['pass_exam'].append(log_var.get('pass_exam', False))

        # log_dict['batch_size'] = batch_size
        # log_dict['iter'] = self.cur_iter
        # log_dict['invalid0'] = self.cnt_invalid_0
        # log_dict['invalid1'] = self.cnt_invalid_1
        # log_dict['invalid2'] = self.cnt_invalid_2
        # log_dict['valid'] = self.cnt_valid
        # log_dict['valid2'] = self.cnt_valid_2
        # log_dict['iou_scores'] = 0 if len(self.iou_scores) == 0 else sum(self.iou_scores)/len(self.iou_scores)


        return losses, log_dict

    def cal_loss_dqn(self, games):

        # states = torch.stack([game['cur_state'] for game in games], dim=0)
        point_feats = torch.stack([game['point_feats'] for game in games], dim=0)
        point_preds = torch.stack([game['point_preds'] for game in games], dim=0)
        point_preds_ori = torch.stack([game['point_preds_ori'] for game in games], dim=0)
        gt_edges = [game['gt_edges'] for game in games]
        point_seqs = torch.stack([game['point_seqs'] for game in games], dim=0)
        gt_polygons = [game['gt_polygons'] for game in games]
        gt_poly_idxes_list = [game['poly_idxes_list'] for game in games]

        scores = self.forward_gnn(point_preds, point_feats, point_seqs, type='act')
        actions = self.sample_actions(scores, point_seqs, method=self.action_sample_type)
        # actions = torch.stack(actions, dim=0)
        # next_states, rewards, terminals, log_vars = [], [], [], []

        next_point_seqs, rewards, terminals, log_vars = self.get_next_states(
            point_seqs, actions, point_preds_ori, gt_polygons, gt_poly_idxes_list
        )

        next_scores = self.forward_gnn(point_preds, point_feats, next_point_seqs, type='target')

        Q_batch = torch.gather(scores, 1, actions.unsqueeze(1)).view(-1)
        Y_batch = torch.zeros_like(Q_batch)
        Y_batch[~terminals] = rewards[~terminals] + next_scores.max(dim=1)[0][~terminals] * self.gamma
        Y_batch[terminals] = rewards[terminals]

        # Y_batch = torch.tensor([
        #     reward if terminal else \
        #     reward + self.gamma * next_pred.max().item() \
        #     for reward, terminal, next_pred in zip(rewards, terminals, next_scores)
        # ], device=scores.device)

        pos_mask = Y_batch > 0
        loss_dqn = self.loss_fun(Q_batch, Y_batch)

        # weights = torch.ones_like(loss_dqn)
        # weights[terminal_mask] = self.terminal_weight
        # weights[terminal_mask & pos_mask] = self.pos_terminal_weight

        # weights = weights / weights.sum()
        # loss_dqn = (loss_dqn * weights).mean()
        loss_terminal = loss_dqn[terminals].sum() / (terminals.sum() + 1e-8)
        loss_dqn = loss_dqn[~terminals].sum() / ((~terminals).sum() + 1e-8)

        for i, game in enumerate(games):
            game['point_seqs'] = next_point_seqs[i]

        return {'loss_dqn': loss_dqn, 'loss_terminal': loss_terminal}, log_vars

    def gen_gt_point_seqs(self, gt_poly_idxes_list, batch_idxes, batch_sizes, device):
        def cycle_shuffle(lst):
            n = len(lst)
            shift = random.randint(0, n - 1)
            lst = lst[-shift:] + lst[:-shift]
            return lst

        B = len(batch_sizes)
        batch_lens = [len(x) for x in batch_idxes]
        num_comp = sum(batch_lens)

        # gt_point_seqs = torch.zeros(num_comp, self.num_max_points, device=device, dtype=torch.long) - 1
        batch_actions = []
        batch_gt_point_seqs = []
        for i in range(B):
            actions = []
            gt_point_seqs = []
            start_idx = 0
            for j, idx in enumerate(batch_idxes[i]):
                cur_gt_point_seqs = torch.zeros(self.num_max_points, device=device, dtype=torch.long) - 1
                N = batch_sizes[i][j]
                gt_poly_idxes = gt_poly_idxes_list[idx]
                if len(gt_poly_idxes) == 0:
                    actions.append(N)
                    continue

                shuffled_idxes = [cycle_shuffle(x[:-1]) for x in gt_poly_idxes]
                shuffled_idxes = [x + x[0:1] for x in shuffled_idxes]
                random.shuffle(shuffled_idxes)
                shuffled_idxes.append([N])
                shuffled_idxes = [torch.tensor(x, device=device) for x in shuffled_idxes]
                shuffled_idxes = torch.cat(shuffled_idxes, dim=0)
                # if len(gt_poly_idxes) > 1:
                #     pdb.set_trace()

                assert len(shuffled_idxes) > 1 # include terminate node

                len_seq = random.randint(0, len(shuffled_idxes)-1)
                cur_gt_point_seqs[:len_seq] = shuffled_idxes[:len_seq]
                actions.append(shuffled_idxes[len_seq].item())
                gt_point_seqs.append(cur_gt_point_seqs)

                start_idx += N

            batch_actions.append(actions)
            batch_gt_point_seqs.append(gt_point_seqs)

        return batch_gt_point_seqs, batch_actions

    def cal_loss_gt_state(self, games):

        points = games['points']
        points_ori = games['points_ori']
        point_feats = games['point_feats']
        batch_sizes = games['batch_sizes']
        batch_idxes = games['batch_idxes']
        gt_poly_idxes_list = games['poly_idxes_list']

        B, N, _ = points.shape
        gt_point_seqs, batch_actions = self.gen_gt_point_seqs(
            gt_poly_idxes_list, batch_idxes, batch_sizes, device=points[0].device
        )

        scores = self.forward_gnn(points, point_feats, gt_point_seqs, batch_sizes, type='act')
        gt_scores = torch.zeros_like(scores)
        attn_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, actions in enumerate(batch_actions):
            start_idx = 0
            for j, action in enumerate(actions):
                gt_scores[i, start_idx+action] = 1
                # start_idx += batch_sizes[i][j]
                start_idx += batch_sizes[i][j] + 1

        for i, sizes in enumerate(batch_sizes):
            start_idx = 0
            for size in sizes:
                attn_mask[i, start_idx:start_idx+size+1] = 1
                start_idx += size + 1

        pos_mask = gt_scores[attn_mask] > 0.5
        loss_gt_state = self.loss_fun(scores, gt_scores)[attn_mask]

        loss_gt_state_pos = loss_gt_state[pos_mask].mean()
        loss_gt_state_neg = loss_gt_state[~pos_mask].mean()
        losses = dict(
            loss_gt_state_pos=loss_gt_state_pos * self.loss_gt_state_pos_weight,
            loss_gt_state_neg=loss_gt_state_neg * self.loss_gt_state_neg_weight
        )

        return losses


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
            return self.forward_test(**kwargs)


