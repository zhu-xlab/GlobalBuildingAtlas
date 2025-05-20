import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

import os

from .parts.miniViT import mViT, mViTHTC
from .unet import Up, Down, DoubleConv, OutConv

class _UNet(nn.Module):
    def __init__(self, num_classes, fusion_mode):
        super(_UNet, self).__init__()
        self.n_channels = 3
        self.bilinear = True
        self.fusion_mode = fusion_mode

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 3
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.out = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_o1 = self.up1(x5, x4)
        x_o2 = self.up2(x_o1, x3)
        x_o3 = self.up3(x_o2, x2)
        x_o4 = self.up4(x_o3, x1)
        out = self.out(x_o4)
        if self.fusion_mode == 'single':
            return [out]
        elif self.fusion_mode == 'first':
            return [x_o2, x_o3, out]
        elif self.fusion_mode == 'second':
            return [x_o2, x_o4, out]
        elif self.fusion_mode == 'third':
            return [x_o2, x_o3, x_o4, out]
        elif self.fusion_mode == 'last':
            return [x_o3, x_o4, out]
        else:
            NotImplementedError

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, ind=5, original_size=False, fusion_mode='normal'):
        super(DecoderBN, self).__init__()
        self.original_size = original_size
        self.fusion_mode = fusion_mode
        bottleneck_features_list = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        up1_features_list = [112, 112, 120, 136, 160, 176, 200, 224]
        up2_features_list = [40, 40, 48, 48, 56, 64, 72, 80]
        up3_features_list = [24, 24, 24, 32, 32, 40, 40, 48]
        up4_features_list = [16, 16, 16, 24, 24, 24, 32, 32]
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features_list[ind], features, kernel_size=1, stride=1) # There should be no paddings here.

        self.up1 = UpSampleBN(skip_input=features // 1 + up1_features_list[ind], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + up2_features_list[ind], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + up3_features_list[ind], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + up4_features_list[ind], output_features=features // 16)
        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3) # 1024 x  16 x  16
        x_d2 = self.up2(x_d1, x_block2) #  512 x  32 x  32
        x_d3 = self.up3(x_d2, x_block1) #  256 x  64 x  64
        x_d4 = self.up4(x_d3, x_block0) #  128 x 128 x 128
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        if self.fusion_mode == 'single':
            return [out]
        elif self.fusion_mode == 'first':
            return [x_d2, x_d3, out]
        elif self.fusion_mode == 'second':
            return [x_d2, x_d4, out]
        elif self.fusion_mode == 'third':
            return [x_d2, x_d3, x_d4, out]
        elif self.fusion_mode == 'last':
            return [x_d3, x_d4, out]
        else:
            NotImplementedError

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks') | (k == 'features'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class MultiLevelUnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=256, min_val=1e-8, max_val=309, backbone_name='efficientnetb5', adanum=False, head_tail_cut=False, earlier=False, fusion_mode='normal', patch_size=4, n_query_channel=128, htc_source="pred"):
        super(MultiLevelUnetAdaptiveBins, self).__init__()
        self.htc_source = htc_source
        self.head_tail_cut = head_tail_cut
        self.earlier = earlier
        self.fusion_mode = fusion_mode
        self.patch_size=patch_size

        self.max_val, self.min_val = max_val, min_val
        if self.fusion_mode == 'single':
            feature_channels = [128]
            n_query_channels = [128]
            conv_out_channels = [n_bins]
        elif self.fusion_mode == 'first':
            feature_channels = [512, 256, 128]
            n_query_channels = [16, 32, n_query_channel]
            conv_out_channels = [32, 64, n_bins]
        elif self.fusion_mode == 'second':
            feature_channels = [512, 128, 128]
            n_query_channels = [16, n_query_channel, n_query_channel]
            conv_out_channels = [32, n_bins, n_bins]
        elif self.fusion_mode == 'third':
            feature_channels = [512, 256, 128, 128]
            n_query_channels = [16, 32, n_query_channel, n_query_channel]
            conv_out_channels = [32, 64, n_bins, n_bins]
        elif self.fusion_mode == 'last':
            feature_channels = [256, 128, 128]
            n_query_channels = [32, n_query_channel, n_query_channel]
            conv_out_channels = [64, n_bins, n_bins]
        else:
            NotImplementedError

        if backbone_name == "unet":
            feature_channels = [128, 64, 64, 128]
            n_query_channels = [16, 32, n_query_channel, n_query_channel]
            conv_out_channels = [32, 64, n_bins, n_bins]
        
        if backbone_name.startswith('efficientnet'):
            self.encoder = Encoder(backend)
            self.decoder = DecoderBN(num_classes=128, ind=int(backbone_name[-1]), fusion_mode=fusion_mode)
        elif backbone_name == "unet":
            self.encoder = nn.Identity()
            self.decoder = backend
        self.adaptive_bins_layers = nn.ModuleList()
        self.convs_out = nn.ModuleList()

        self.convs_out_bg = nn.ModuleList() if self.head_tail_cut else None
        self.convs_htc = nn.ModuleList() if self.head_tail_cut else None
        vit = mViTHTC if self.earlier else mViT

        for (fc, nqc, coc) in zip(feature_channels, n_query_channels, conv_out_channels):
            self.adaptive_bins_layers.append(vit(fc, n_query_channels=nqc, patch_size=self.patch_size, dim_out=coc, embedding_dim=128, norm='softmax')) # patch size changed from 8 to 4.
            conv_out = nn.Sequential(
                nn.Conv2d(nqc, coc, 1, 1, 0),
                nn.Softmax(dim=1)
            )
            self.convs_out.append(conv_out)
            if self.head_tail_cut:
                conv_htc = nn.Sequential(
                    nn.Conv2d(nqc, 1, 1, 1, 0),
                    nn.Sigmoid()
                )
                self.convs_htc.append(conv_htc)
                if self.earlier:
                    conv_out_bg = nn.Sequential(
                        nn.Conv2d(nqc, coc, 1, 1, 0),
                        nn.Softmax(dim=1)
                    )
                else:
                    conv_out_bg = nn.Sequential(
                        nn.Conv2d(nqc, nqc*2, 3, 1, 1),
                        nn.BatchNorm2d(nqc*2),
                        nn.LeakyReLU(),
                        nn.Conv2d(nqc*2, nqc, 1, 1, 0),
                        nn.BatchNorm2d(nqc),
                        nn.LeakyReLU(),
                        nn.Conv2d(nqc, 1, 1, 1, 0)
                    )
                self.convs_out_bg.append(conv_out_bg)
                
    def forward(self, x, gt=None):
        '''
        x: features from DecoderBN, x_d2 (512 x 32 x 32), x_d3 (256 x 64 x 64), and out (128 x 128 x 128) are taken here.
        '''
        if self.htc_source != "pred":
            assert gt is not None, "GT should be available when htc is not predicted."
        x = self.decoder(self.encoder(x))
        bin_edges_list = []
        pred_list = []
        out_list = []
        centers_list = []
        if self.head_tail_cut:
            htc_prob_list = []
        for i, xx in enumerate(x):
            if self.head_tail_cut & self.earlier:
                bin_widths_normed, range_attention_maps, range_attention_maps_bg = self.adaptive_bins_layers[i](xx)
            else:
                bin_widths_normed, range_attention_maps = self.adaptive_bins_layers[i](xx)
            out = self.convs_out[i](range_attention_maps)
            if self.head_tail_cut:
                if self.htc_source == "pred":
                    out_htc = self.convs_htc[i](range_attention_maps)
                    fg_mask = (out_htc<0.5).to(torch.float)
                elif self.htc_source == "bf":
                    assert "mask" in gt, "Building footprints should be available!"
                    out_htc = F.interpolate(gt["mask"], size=out.shape[2:], mode="nearest")
                    fg_mask = (out_htc>0).to(torch.float)
                htc_prob_list.append(out_htc)
                if self.earlier:
                    out_bg = self.convs_out_bg[i](range_attention_maps_bg)
                else:
                    pred_bg = self.convs_out_bg[i](range_attention_maps)


            bin_widths = (self.max_val - self.min_val) * bin_widths_normed
            bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
            bin_edges = torch.cumsum(bin_widths, dim=1)
            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
            if self.head_tail_cut & self.earlier:
                out = fg_mask * out + (1-fg_mask) * out_bg
            pred = torch.sum(out * centers[:, :, None, None], dim=1, keepdim=True)
            if self.head_tail_cut & (not self.earlier):
                pred = torch.where(fg_mask>0, torch.clamp(pred, min=1), torch.clamp(pred_bg, max=1))
            
            bin_edges_list.append(bin_edges)
            pred_list.append(pred)
            out_list.append(out)
            centers_list.append(centers)
        if self.head_tail_cut:
            return bin_edges_list, pred_list, out_list, centers_list, htc_prob_list
        else:
            return bin_edges_list, pred_list, out_list, centers_list

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self, min_val=1e-8):
        super().__init__()
        self.name = "ChamferLoss"
        self.min_val = min_val

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape
        bin_lengths = None
        
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(self.min_val)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, x_lengths=bin_lengths, y_lengths=target_lengths)
            
        return loss

class UBins(nn.Module):
    def __init__(self, cfgs):
        super(UBins, self).__init__()
        self.adanum = False
        ## multi level fusion.
        self.fusion_mode = cfgs.get("fusion_mode", 'last')

        ## head tail cut.
        self.head_tail_cut = cfgs.get("head_tail_cut", False)
        self.htc_thres = cfgs.get("htc_thres", 1)
        self.earlier = cfgs.get("earlier", False)
        self.htc_source = cfgs.get("htc_source", "pred")

        ## probability distribution constraint.
        self.loss_p = cfgs.get("prob_loss", False)
        self.loss_p_bg = cfgs.get("prob_loss_bg", False)

        self.num_bins = cfgs.get("num_classes", 256)
        self.patch_size = cfgs.get("patch_size", 4)

        data_dir = cfgs.get("data_dir", "data/gbh/")
        ndsm_stats_file = os.path.join(data_dir, "ndsm_stats.pickle")
        _, _, _, self.h_max, _ = torch.load(ndsm_stats_file)
        self.h_min = self.htc_thres if (self.head_tail_cut & (not self.earlier)) else 1e-8
        self.n_query_channel = cfgs.get("n_query_channel", self.num_bins)

        kwargs = dict(
            n_bins=self.num_bins, 
            min_val=self.h_min, 
            max_val=self.h_max, 
            head_tail_cut=self.head_tail_cut, 
            earlier=self.earlier, 
            fusion_mode=self.fusion_mode,
            patch_size=self.patch_size,
            n_query_channel=self.n_query_channel,
            htc_source=self.htc_source)
        
        self.backbone_name = cfgs.get("backbone", "efficientnetb0")
        if self.backbone_name.startswith("efficientnetb"):
            ind = self.backbone_name[-1]
            basemodel_name = 'tf_efficientnet_b'+ind+'_ap'
            if os.path.exists('rwightman_gen-efficientnet-pytorch_master'):
                basemodel = torch.hub.load('rwightman_gen-efficientnet-pytorch_master', basemodel_name, source='local', pretrained=True)
            else:
                basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
        elif self.backbone_name == "unet":
            basemodel = _UNet(num_classes=128, fusion_mode=self.fusion_mode)
        self.model = MultiLevelUnetAdaptiveBins(backend=basemodel, backbone_name=self.backbone_name, **kwargs)

        self.binloss = BinsChamferLoss(min_val=self.h_min)
        self.chamfer_w = cfgs.get("chamfer_weight", 0.01)
        self.htc_w = cfgs.get("htc_weight", 1)
        self.dc_w = cfgs.get("dc_weight", 1)

    def forward(self, x, bf=None):
        gt={"mask": bf}
        if not self.head_tail_cut:
            bins, out_raw, prob, centers = self.model(x)
        else:
            bins, out_raw, prob, centers, htc_prob = self.model(x, gt)
        return F.interpolate(out_raw[-1], size=x.shape[2:], mode='bilinear')
