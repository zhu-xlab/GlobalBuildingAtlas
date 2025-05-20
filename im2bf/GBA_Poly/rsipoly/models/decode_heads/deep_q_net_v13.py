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

@HEADS.register_module()
class DeepQNetV13(BaseModule):

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
        max_seq_size=256, seq_feat_size=128, max_batch_size=256, max_base_size=80,
        encode_points=False, num_bits=11, seq_feats_type='normal'
    ):
        super(DeepQNetV13, self).__init__()

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
        self.max_seq_size = max_seq_size
        self.seq_feat_size = seq_feat_size
        self.max_batch_size = max_batch_size
        self.max_base_size = max_base_size
        self.encode_points = encode_points
        self.num_bits = num_bits
        self.seq_feats_type = seq_feats_type

    def greedy_arange(self, sizes, max_base_size, plus_one=False):
        return self.random_arange(sizes, max_base_size, plus_one=plus_one)

    def random_arange(self, sizes, max_base_size, plus_one=False):

        base_size = max_base_size
        batch_idx_list = []
        batch_size_list = []
        rand_permute = np.random.permutation(len(sizes))
        cur_bin = []
        cur_sizes = []
        cur_size = 0
        for i, idx in enumerate(rand_permute):
            if sizes[idx] < 4:
                continue
            if sizes[idx] + plus_one >= base_size:
                continue
            if cur_size + sizes[idx] + plus_one < base_size:
                cur_bin.append(idx)
                cur_sizes.append(sizes[idx])
                cur_size += sizes[idx] + plus_one
            else:
                # finish the current bin
                batch_idx_list.append(cur_bin)
                batch_size_list.append(cur_sizes)
                cur_bin = [idx]
                cur_size = sizes[idx]
                cur_sizes = [sizes[idx]]

        # deal with the remaining items
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

    def sample_games(self, level=None, add_dummy=False):

        ## TODO: improve the greedy_arange algorithm
        sizes = np.array([len(x) for x in self.point_feats_list])
        batch_idx_list, batch_size_list = self.random_arange(sizes, self.max_base_size, add_dummy)

        if len(batch_idx_list) == 0:
            return None

        num_limit_points = 512 * 512 * 32
        # base_size = sizes[batch_idx_list[0][0]]
        base_size = self.max_base_size
        num_max_batch = num_limit_points // base_size ** 2

        ## TODO: shuffle the list
        if len(batch_idx_list) > num_max_batch:
            batch_idx_list = batch_idx_list[:num_max_batch]
            batch_size_list = batch_size_list[:num_max_batch]


        device = self.point_feats_list[0].device
        # batch_size = min(num_limit_points // (game_level**2), len(self.game_pool[game_level]))
        # batch_size = min(batch_size, self.max_batch_size)

        new_batch_size_list = []
        merged_batch_size_list = []
        new_batch_idx_list = []
        # merged_batch_gt_polygons = []
        # merged_batch_poly_idxes = []
        batch_gt_polygons = []
        batch_poly_idxes = []
        for i, batch_idxes in enumerate(batch_idx_list):
            num_gt_nodes = 0
            gt_polygons = []
            gt_poly_idxes = []
            start_idx = 0
            for j, idx in enumerate(batch_idxes):
                gt_poly_sizes = [len(x) for x in self.poly_idxes_list[idx]]
                num_gt_nodes += sum(gt_poly_sizes) + len(gt_poly_sizes)
                gt_polygons += [
                    self.point_preds_ori_list[idx][poly_idxes].tolist() for poly_idxes in self.poly_idxes_list[idx]
                ]
                gt_poly_idxes += [
                    (np.array(poly_idxes) + start_idx).tolist() for poly_idxes in self.poly_idxes_list[idx]
                ]
                start_idx += batch_size_list[i][j]

            if num_gt_nodes >= base_size:
                continue

            batch_gt_polygons.append([gt_polygons])
            batch_poly_idxes.append([gt_poly_idxes])
            new_batch_size_list.append(batch_size_list[i])
            merged_batch_size_list.append([sum(batch_size_list[i])])
            new_batch_idx_list.append(batch_idxes)

        batch_idx_list = new_batch_idx_list

        B = len(new_batch_size_list)
        if B == 0:
            return None

        batch_points = torch.zeros(
            B, base_size, 2, device=device
        )
        batch_points_ori = torch.zeros(
            B, base_size, 2, device=device
        )
        batch_point_feats = torch.zeros(
            B, base_size, self.point_feats_list[0].size(1),
            device=device
        )
        dummy_points = torch.zeros(1, 2, device=device) - 1
        dummy_feats = torch.zeros(1, self.point_feats_list[0].size(1), device=device) - 1

        for i, batch_idxes in enumerate(batch_idx_list):
            if add_dummy:
                cur_batch_points = torch.cat([torch.cat([self.point_preds_list[idx], dummy_points]) for idx in batch_idxes])
                cur_batch_points_ori = torch.cat([torch.cat([self.point_preds_ori_list[idx], dummy_points]) for idx in batch_idxes])
                cur_batch_point_feats = torch.cat([torch.cat([self.point_feats_list[idx], dummy_feats]) for idx in batch_idxes])
            else:
                cur_batch_points = torch.cat([self.point_preds_list[idx] for idx in batch_idxes])
                cur_batch_points_ori = torch.cat([self.point_preds_ori_list[idx] for idx in batch_idxes])
                cur_batch_point_feats = torch.cat([self.point_feats_list[idx] for idx in batch_idxes])

            assert len(cur_batch_points) <= base_size
            batch_points[i, :len(cur_batch_points)] = cur_batch_points
            batch_points_ori[i, :len(cur_batch_points_ori)] = cur_batch_points_ori
            batch_point_feats[i, :len(cur_batch_point_feats)] = cur_batch_point_feats

        games = dict(
            points=batch_points,
            points_ori=batch_points_ori,
            point_feats=batch_point_feats,
            batch_sizes=merged_batch_size_list,
            ori_batch_sizes=new_batch_size_list,
            # poly_idxes_list=self.poly_idxes_list,
            batch_gt_polygons=batch_gt_polygons,
            batch_poly_idxes=batch_poly_idxes
        )

        return games

    def update_target_net(self):
        self.target_net.load_state_dict(self.act_net.state_dict())

    def update_act_net(self):
        self.act_net.load_state_dict(self.target_net.state_dict())

    def encode_to_binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


    def forward_gnn(self, point_preds, points_ori, point_feats, point_seqs, batch_sizes, type='act',
                    plus_one=False):

        B, N, _ = point_preds.shape
        C = point_feats.shape[-1]
        T = self.seq_feat_size
        T2 = self.max_seq_size
        # if point_seqs.max() >= N:
        #     pdb.set_trace()

        points_enc = self.encode_to_binary(
            points_ori.view(-1).long(), self.num_bits
        ).view(B, N, 2 * self.num_bits)

        point_seq_feats = torch.zeros(B, T, N, device=point_preds.device, dtype=point_preds.dtype)
        attn_mask = ~torch.eye(N+1, N+1, device=point_preds.device, dtype=torch.bool)
        attn_mask = attn_mask.view(1, N+1, N+1).repeat(B, 1, 1)

        for idx in range(B):
            start_idx = 0
            # assert len(batch_sizes[idx]) == 1
            for idx2, size in enumerate(batch_sizes[idx]):
                seqs = point_seqs[idx][idx2]
                seq_idxes = (seqs >= 0).nonzero().view(-1)

                temp = torch.bincount(seqs[seqs>=0])
                degrees = torch.zeros(T2, device=seqs.device, dtype=torch.long)
                degrees[:len(temp)] = temp
                degrees = torch.where(seqs >= 0, degrees[seqs], 0)

                arange = torch.arange(T2, device=seqs.device, dtype=torch.long)
                cur_poly_start_idx = torch.where(degrees == 2, arange + 1, 0).max()
                cur_poly_end_idx = len(seq_idxes)

                if self.seq_feats_type == 'no_order':
                    # 0: the front_node
                    # 1: nodes of the current polygon
                    # 2: finished nodes
                    if cur_poly_end_idx > 0 and not cur_poly_start_idx == cur_poly_end_idx:
                        point_seq_feats[
                            idx, 0,
                            seqs[cur_poly_end_idx - 1] + start_idx
                        ] = 1
                    point_seq_feats[
                        idx, 1, seqs[cur_poly_start_idx:cur_poly_end_idx] + start_idx
                    ] = 1
                    point_seq_feats[
                        idx, 2, seqs[:cur_poly_start_idx] + start_idx
                    ] = 1

                else:
                    assert seq_idxes.shape[0] < T - 1
                    point_seq_feats[
                        idx, seq_idxes[cur_poly_start_idx:] - cur_poly_start_idx,
                        seqs[seq_idxes[cur_poly_start_idx:]] + start_idx
                    ] = 1
                    point_seq_feats[
                        idx, -1, # location -1 indicate ended polygons
                        seqs[seq_idxes[:cur_poly_start_idx]] + start_idx
                    ] = 1

                if plus_one:
                    attn_mask[idx, start_idx:start_idx+size+1, start_idx:start_idx+size+1] = 0
                else:
                    attn_mask[idx, start_idx:start_idx+size, N] = 0
                    attn_mask[idx, N, start_idx:start_idx+size] = 0
                    attn_mask[idx, N, N] = 0

                start_idx += size + plus_one

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.act_net.num_heads, 1, 1).view(-1, N+1, N+1)

        # seq_idxes = (point_seqs >= 0).nonzero()
        # if not (len(seq_idxes) == 0 or seq_idxes[:,1].max() < T):
        #     pdb.set_trace()
        # point_seq_feats[seq_idxes[:,0], seq_idxes[:,1], point_seqs[seq_idxes[:,0], seq_idxes[:,1]]] = 1

        if self.encode_points:
            node_feats = torch.cat([points_enc.permute(0,2,1), point_feats.permute(0,2,1), point_seq_feats], dim=1)
            dim_point = 11 * 2
        else:
            node_feats = torch.cat([point_preds.permute(0,2,1), point_feats.permute(0,2,1), point_seq_feats], dim=1)
            dim_point = 2

        terminate_feats = torch.zeros(1, T + dim_point + C, 1, device=point_preds.device) - 1
        terminate_feats = terminate_feats.repeat(B, 1, 1)
        node_feats = torch.cat([node_feats, terminate_feats], dim=2)

        if type == 'act':
            scores = self.act_net(node_feats, attn_mask=attn_mask)
        else:
            scores = self.target_net(node_feats, attn_mask=attn_mask)

        scores = scores.view(B, N+1)
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

                    if cur_poly_end_idx >= self.seq_feat_size - 2:
                        # exceed lenth limit of the sequence
                        action = batch_sizes[i][j]

                    else:

                        valid_mask = torch.ones(batch_sizes[i][j]+1, device=seqs.device, dtype=torch.bool)
                        invalid_seq_idxes = seqs[:cur_poly_end_idx]
                        valid_seq_idxes = seqs[cur_poly_start_idx] if seqs[cur_poly_start_idx] >= 0 else None

                        valid_mask[invalid_seq_idxes] = 0
                        if valid_seq_idxes is not None and cur_poly_start_idx + 4 <= cur_poly_end_idx:
                            valid_mask[valid_seq_idxes] = 1


                        if cur_poly_start_idx == 0 or (not cur_poly_start_idx == cur_poly_end_idx and valid_mask.sum() > 1):
                            valid_mask[-1] = False

                        assert valid_mask.any()
                        cur_scores = scores[i, valid_mask.nonzero().view(-1) + start_idx]
                        action = cur_scores.argmax()
                        action = valid_mask.nonzero().view(-1)[action].item()
                        # if (seqs == action).any():
                        #     pdb.set_trace()
                    actions.append(action)
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

        assert len(point_seqs) == B
        assert len(actions) == B
        if gt_polygons is not None:
            assert len(gt_polygons) == B

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
                        [Polygon(x) for x in gt_polygons[i][j]] if gt_polygons is not None else None
                    )

                    if not N in ious:
                        ious[N] = []
                        return_polygons[N] = []
                    ious[N].append(reward)
                    return_polygons[N].extend(valid_polygons)


                rewards.append(reward)
                offset += N


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
        # batch_sizes = games['ori_batch_sizes']
        batch_gt_polygons = games.get('batch_gt_polygons', None)

        B, N, _ = points.shape


        active_games = torch.ones(B, device=points.device, dtype=torch.bool)
        cur_point_seqs = []
        success_vector = np.zeros(B, dtype=np.int)

        for i in range(B):
            point_seqs = []
            sizes = []
            for j, size in enumerate(batch_sizes[i]):
                point_seqs.append(torch.zeros(self.max_seq_size, dtype=torch.long,
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

            scores = torch.zeros(B, N + 1, device=points.device)
            with torch.no_grad():
                active_scores = self.forward_gnn(
                    points[active_games],
                    points_ori[active_games],
                    point_feats[active_games],
                    [cur_point_seqs[x] for x in game_idx_list],
                    cur_batch_sizes,
                    type=net_type,
                    plus_one=True
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

        pdb.set_trace()
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

        gt_polygons = []
        for polygons in batch_gt_polygons:
            gt_polygons += [torch.tensor(x) for x in polygons[0]]

        return dict(
            success_rate=success_rate,
            success_iou=success_iou,
            success_cnt=success_cnt,
            all_cnt=all_cnt,
            return_polygons=return_polygons,
            gt_polygons = gt_polygons
        )

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

    def gen_gt_point_seqs(self, batch_poly_idxes, batch_sizes, device):
        def cycle_shuffle(lst):
            n = len(lst)
            shift = random.randint(0, n - 1)
            lst = lst[-shift:] + lst[:-shift]
            return lst

        B = len(batch_sizes)
        batch_actions = []
        batch_gt_point_seqs = []
        new_batch_sizes = []

        for i in range(B):
            actions = []
            gt_point_seqs = []
            new_sizes = []

            start_idx = 0
            assert len(batch_sizes[i]) == 1
            for j, _ in enumerate(batch_sizes[i]):
                cur_gt_point_seqs = torch.zeros(self.max_seq_size, device=device, dtype=torch.long) - 1
                N = batch_sizes[i][j]
                new_sizes.append(N)
                gt_poly_idxes = batch_poly_idxes[i][j]

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
            new_batch_sizes.append(new_sizes)

        return batch_gt_point_seqs, batch_actions, new_batch_sizes



    def cal_loss_gt_state(self, games):

        points = games['points']
        points_ori = games['points_ori']
        point_feats = games['point_feats']
        batch_sizes = games['batch_sizes']
        batch_poly_idxes = games['batch_poly_idxes']

        B, N, _ = points.shape
        gt_point_seqs, batch_actions, new_batch_sizes = self.gen_gt_point_seqs(
            batch_poly_idxes, batch_sizes, device=points[0].device
        )

        scores = self.forward_gnn(points, points_ori, point_feats, gt_point_seqs, new_batch_sizes, type='act')
        gt_scores = torch.zeros_like(scores)
        attn_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, actions in enumerate(batch_actions):
            start_idx = 0
            for j, action in enumerate(actions):
                gt_scores[i, start_idx+action] = 1
                # start_idx += batch_sizes[i][j]
                start_idx += new_batch_sizes[i][j]

        for i, sizes in enumerate(new_batch_sizes):
            start_idx = 0
            for j, size in enumerate(sizes):
                seqs = gt_point_seqs[i][j]
                T = len(seqs)
                temp = torch.bincount(seqs[seqs>=0])
                degrees = torch.zeros(T, device=seqs.device, dtype=torch.long)
                degrees[:len(temp)] = temp
                degrees = torch.where(seqs >= 0, degrees[seqs], 0)

                arange = torch.arange(T, device=seqs.device, dtype=torch.long)
                cur_poly_start_idx = torch.where(degrees == 2, arange + 1, 0).max()
                cur_poly_end_idx = (seqs >= 0).sum()

                if cur_poly_start_idx == cur_poly_end_idx and cur_poly_start_idx > 0:
                    attn_mask[i, start_idx:start_idx+size+1] = 1
                else:
                    attn_mask[i, start_idx:start_idx+size] = 1

                start_idx += size

        pos_mask = gt_scores[attn_mask] > 0.5
        loss_gt_state = self.loss_fun(scores, gt_scores)[attn_mask]

        loss_gt_state_pos = loss_gt_state[pos_mask].mean()
        loss_gt_state_neg = loss_gt_state[~pos_mask].mean()
        losses = dict(
            loss_gt_state_pos=loss_gt_state_pos * self.loss_gt_state_pos_weight,
            loss_gt_state_neg=loss_gt_state_neg * self.loss_gt_state_neg_weight
        )

        return losses

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
