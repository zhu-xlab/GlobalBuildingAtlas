import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
from dataloaders.custom_transforms import denormalizeimage
import time
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
import pickle


class DenseCRFLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        
        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation=grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None
    

class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        return self.weight*DenseCRFLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
