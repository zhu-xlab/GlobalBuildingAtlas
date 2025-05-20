# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
import warnings
import pdb

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook
from mmcv.utils import digit_version

from rsipoly.ops import resize
from rsipoly.core import DistEvalHook, EvalHook
# from rsipoly.core.mask.structures import polygon_to_bitmap


@HOOKS.register_module()
class WandbHookSeg(WandbLoggerHook):
    """Enhanced Wandb logger hook for MMDetection.

    Comparing with the :cls:`mmcv.runner.WandbLoggerHook`, this hook can not
    only automatically log all the metrics but also log the following extra
    information - saves model checkpoints as W&B Artifact, and
    logs model prediction as interactive W&B Tables.

    - Metrics: The MMDetWandbHook will automatically log training
        and validation metrics along with system metrics (CPU/GPU).

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
        every checkpoint interval will be saved as W&B Artifacts.
        This depends on the : class:`mmcv.runner.CheckpointHook` whose priority
        is higher than this hook. Please refer to
        https://docs.wandb.ai/guides/artifacts/model-versioning
        to learn more about model versioning with W&B Artifacts.

    - Checkpoint Metadata: If evaluation results are available for a given
        checkpoint artifact, it will have a metadata associated with it.
        The metadata contains the evaluation metrics computed on validation
        data with that checkpoint along with the current epoch. It depends
        on `EvalHook` whose priority is more than MMDetWandbHook.

    - Evaluation: At every evaluation interval, the `MMDetWandbHook` logs the
        model prediction as interactive W&B Tables. The number of samples
        logged is given by `num_eval_images`. Currently, the `MMDetWandbHook`
        logs the predicted bounding boxes along with the ground truth at every
        evaluation interval. This depends on the `EvalHook` whose priority is
        more than `MMDetWandbHook`. Also note that the data is just logged once
        and subsequent evaluation tables uses reference to the logged data
        to save memory usage. Please refer to
        https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    For more details check out W&B's MMDetection docs:
    https://docs.wandb.ai/guides/integrations/rsidetection

    ```
    Example:
        log_config = dict(
            ...
            hooks=[
                ...,
                dict(type='MMDetWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     interval=50,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100,
                     bbox_score_thr=0.3)
            ])
    ```

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations). Defaults to 50.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint. Defaults to False.
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Defaults to True.
        num_eval_images (int): The number of validation images to be logged.
            If zero, the evaluation won't be logged. Defaults to 100.
        bbox_score_thr (float): Threshold for bounding box scores.
            Defaults to 0.3.
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 num_eval_images=10,
                 **kwargs):
        super(WandbHookSeg, self).__init__(init_kwargs, interval, **kwargs)

        self.num_eval_images = num_eval_images

    @master_only
    def before_run(self, runner):
        super(WandbHookSeg, self).before_run(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

    @master_only
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(WandbHookSeg, self).after_train_iter(runner)
        else:
            super(WandbHookSeg, self).after_train_iter(runner)

        # if self.by_epoch:
        #     return

        if self.every_n_iters(runner, self.interval):
        # if True:
            # results = self.eval_hook.latest_results
            results = runner.outputs
            if 'states' in results:
                self._visualize_train_data(runner, results['states'])


    def _visualize_train_data(self, runner, vis_data):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        CLASSES = model.CLASSES
        class_labels = {}
        for i, class_name in enumerate(CLASSES):
            class_labels[i] = class_name

        img_log = {'gt': [], 'pred': []}
        for i in range(vis_data['img'].shape[0]):
            vis_data['seg_logits'] = resize(
                input=vis_data['seg_logits'],
                size=vis_data['gt'].shape[2:],
                mode='bilinear',
                align_corners=False)

            img = vis_data['img'][i].permute(1,2,0).cpu().numpy()
            gt = vis_data['gt'][i].squeeze(0).cpu().numpy()
            pred = vis_data['seg_logits'][i].argmax(dim=0).cpu().numpy()


            gt_masks = {
                'mask_data': gt,
                'class_labels': class_labels}
            pred_masks = {
                'mask_data': pred,
                'class_labels': class_labels}
            img_log['gt'].append(self.wandb.Image(img, masks={'ground_truth': gt_masks}))
            img_log['pred'].append(self.wandb.Image(img, masks={'predition': pred_masks}))

        self.wandb.log(img_log)

    @master_only
    def after_train_epoch(self, runner):
        pdb.set_trace()
        super(WandbHookSeg, self).after_train_epoch(runner)
        pass

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

