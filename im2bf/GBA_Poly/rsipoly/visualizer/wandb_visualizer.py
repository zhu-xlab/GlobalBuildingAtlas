# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
import warnings
import pdb
from PIL import Image
import wandb
import torch
import os
import cv2
from mmengine.visualization import Visualizer
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook
from mmcv.runner.hooks.logger import LoggerHook
from mmcv.runner import HOOKS
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from .polygon_plot_utils import plot_geometries
import matplotlib.pyplot as plt
import geopandas as gpd

def img_loader(path, retry=5):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    ri = 0
    while ri < retry:
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            ri += 1

@HOOKS.register_module()
class WandbVisualizer(Visualizer, LoggerHook):

    def __init__(self,
                 val_dataset=None,
                 wandb_cfg=None,
                 name='visualizer',
                 **kwargs):

        self.val_dataset = val_dataset
        self.init_cfg = wandb_cfg['init_kwargs']
        self.init_kwargs = self.init_cfg
        self.num_eval_images = wandb_cfg.get('num_eval_images', 0)
        self.bbox_score_thr = wandb_cfg.get('bbox_score_thr', 0.3)
        self.max_polygon_num = wandb_cfg.get('max_polygon_num', 500)
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None
        self.BGR2RGB = wandb_cfg.get('BGR2RGB', True)
        self.without_mask = wandb_cfg.get('without_mask', False)
        self.vis_mask_as_poly = wandb_cfg.get('vis_mask_as_poly', True)
        self.wandb = wandb
        self.interval = wandb_cfg.get('interval', 100)
        self.max_vis_boxes = wandb_cfg.get('max_vis_boxes', 100)
        self.scalar_interval = wandb_cfg.get('scalar_interval', 50)
        self.img_norm_cfg = wandb_cfg.get('img_norm_cfg', None)
        self.wandb.init(
            **self.init_cfg
        )
        self.by_epoch = wandb_cfg.get('by_epoch', False)
        # super(Visualizer, self).__init__(name=name)
        # super(LoggerHook, self).__init__(interval=self.interval, by_epoch=False)
        super(WandbVisualizer, self).__init__(name=name)

        if self.val_dataset is not None and self.num_eval_images > 0:
            # # Initialize data table
            self._init_data_table()
            # # Add data to the data table
            self._add_ground_truth()
            # # Log ground truth data
            self._log_data_table()

    # def before_run(self, runner):
    #     super(WandbVisualizer, self).before_run()

    def process_vis_data(self, states):
        if states is None:
            return None
        if type(states) == dict:
            states = [states]

        for state in states:
            for key, value in state.items():
                if key.startswith('vis|'):
                    if key.split('|')[1].startswith('seg_mask'):
                        self._vis_seg_mask(key.split('|')[1], value)

                    if key.split('|')[1].startswith('scalar'):
                        self._log_scalar(value)

                    elif key.split('|')[1].startswith('density'):
                        self._vis_density(key.split('|')[1], value)

                    elif key.split('|')[1].startswith('hist'):
                        self._vis_hist(key.split('|')[1], value[0], value[1])

                    elif key.split('|')[1].startswith('dets'):
                        self._vis_dets(key.split('|')[1], **value)

                    elif key.split('|')[1].startswith('poly'):
                        self._vis_poly(key.split('|')[1], *value)

                    elif key.split('|')[1].startswith('points'):
                        self._vis_points(key.split('|')[1], *value)

                    elif key.split('|')[1].startswith('featmap'):
                        self._vis_featmap(key.split('|')[1], *value)

                    elif key.split('|')[1].startswith('mask'):
                        self._vis_masks(key.split('|')[1], *value)

                    elif key.split('|')[1].startswith('super_pixel'):
                        self._vis_super_pixel(key.split('|')[1], *value)

                    elif key.split('|')[1].startswith('super-pixel-mask'):
                        self._vis_super_pixel_mask(key.split('|')[1], *value)

    def _result2wandb_image(self, img, bbox_result, segm_result, class_id_to_label=None):

        class_set = self.class_set
        if class_id_to_label is not None:
            class_set = self.wandb.Classes([{
                'id': id,
                'name': name
            } for id, name in class_id_to_label.items()])

        # Get labels
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # Get segmentation mask if available.
        segms = None
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
            assert len(segms) == len(labels)
        # TODO: Panoramic segmentation visualization.

        # Remove bounding boxes and masks with score lower than threshold.
        if type(self.bbox_score_thr) == list or self.bbox_score_thr > 0:
            # assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > np.array(self.bbox_score_thr)[labels]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        # Get dict of bounding boxes to be logged.
        if self.max_vis_boxes is not None:
            bboxes = bboxes[:self.max_vis_boxes]
            labels = labels[:self.max_vis_boxes]
        wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=False, class_id_to_label=class_id_to_label)
        # Get dict of masks to be logged.
        if segms is not None:
            wandb_masks = self._get_wandb_masks(segms, labels)
        else:
            wandb_masks = None

        wandb_img = self.wandb.Image(
            img,
            boxes=wandb_boxes,
            masks=wandb_masks,
            classes=class_set
        )
        return wandb_img

    def _log_scalar(self, data):
        self.wandb.log(data)

    def _vis_dets(self, prefix, imgs, bbox_results, segm_results=None):

        img_log = {}
        img_log[prefix] = []

        for i in range(len(imgs)):

            cur_img = imgs[i]
            if type(cur_img) == torch.Tensor:
                cur_img = cur_img.permute(1,2,0).cpu().numpy()

            if self.BGR2RGB:
                cur_img = cur_img[:, :, [2,1,0]]

            class_id_to_label = None
            if 'proposal' in prefix:
                class_id_to_label = {1: 'proposal', 2: 'background'}

            wandb_img = self._result2wandb_image(
                cur_img, bbox_results[i],
                segm_results[i] if segm_results is not None else None,
                class_id_to_label=class_id_to_label
            )

            # wandb_img = self.wandb.Image(
            #     cur_img,
            #     masks={'mask': vis_mask} if mask is not None else None
            # )
            img_log[prefix].append(wandb_img)

        self.wandb.log(img_log)

    def de_normalize(self, imgs):

        if self.img_norm_cfg is not None:
            if 'vis_bands' in self.img_norm_cfg:
                imgs = imgs[:, self.img_norm_cfg['vis_bands']]
            mean = self.img_norm_cfg['mean']
            std = self.img_norm_cfg['std']
            # imgs *= torch.tensor(std, device=imgs.device).view(1,-1,1,1) / 255.
            # imgs += torch.tensor(mean, device=imgs.device).view(1,-1,1,1) / 255.

            imgs *= torch.tensor(std, device=imgs.device).view(1,-1,1,1)
            imgs += torch.tensor(mean, device=imgs.device).view(1,-1,1,1)

        return imgs


    def _vis_poly(self, prefix, imgs, batch_polygons, alpha=0.5):
        img_log = {}
        img_log[prefix] = []

        imgs = self.de_normalize(imgs)
        B, C, H, W = imgs.shape

        for img, polygons in zip(imgs, batch_polygons):
            img = img.permute(1,2,0).cpu().numpy()

            gdf = gpd.GeoDataFrame({'geometry': polygons})

            num_colors = len(gdf)
            colors = plt.cm.Spectral(np.linspace(0, 1, num_colors))
            np.random.shuffle(colors)
            # gdf['color'] = colors

            ax = gdf.plot(color=colors, edgecolor=colors)

            for i, polygon in enumerate(polygons):
                rings = [polygon.exterior, *polygon.interiors]
                coords = [ring.xy for ring in rings]
                for xi, yi in coords:
                    ax.plot(xi[:-1], yi[:-1], marker="o", color='blue', markersize=20)

            # Customizing plot - removing axes and setting size
            ax.set_axis_off()  # Remove axes
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_position([0, 0, 1, 1])
            ax.invert_yaxis()

            fig = ax.figure
            fig.set_size_inches(100, 100)  # Change the size of the figure

            # plt.tight_layout(pad=0)
            plt.savefig('.temp.png', dpi=9)

            vis_img = cv2.imread('.temp.png')

            if vis_img is not None:
                mask = (vis_img == 255).all(axis=-1)
                # vis_img = cv2.resize(vis_img, img.shape[:2], interpolation=cv2.INTER_NEAREST)
                vis_img = cv2.resize(vis_img, img.shape[:2])
                mask = cv2.resize(mask.astype(np.uint8), img.shape[:2], interpolation=cv2.INTER_NEAREST)

                vis_img = np.where(np.expand_dims(mask, 2), img, vis_img * (1 - alpha) + img * alpha)
                # vis_img = vis_img * (1 - alpha) + img * alpha
                vis_img = vis_img.clip(0, 255).astype(np.uint8)

                img_log[prefix].append(
                    self.wandb.Image(vis_img)
                )

            plt.close()

        self.wandb.log(img_log)

    """
    def _vis_poly(self, prefix, imgs, polygons):
        img_log = {}
        img_log[prefix] = []

        imgs = self.de_normalize(imgs)

        for i in range(len(polygons)):
            cur_polygons = polygons[i]

            if imgs is not None:
                cur_img = imgs[i]
                if type(cur_img) == torch.Tensor:
                    cur_img = cur_img.permute(1,2,0).cpu().numpy()

                self.set_image(cur_img)
            else:
                self.set_image(np.zeros((1024, 1024)))

            if len(cur_polygons) > 0:
                cur_polygons = [torch.tensor(x) for x in cur_polygons]
                self.draw_polygons(cur_polygons, line_widths=1)
            poly_img = self.get_image()
            img_log[prefix].append(
                self.wandb.Image(poly_img)
            )
        self.wandb.log(img_log)
    """

    def _vis_points(self, prefix, imgs, points, point_labels=None):
        img_log = {}
        img_log[prefix] = []

        for i in range(len(imgs)):
            cur_points = points[i]
            cur_img = imgs[i]
            if type(cur_img) == torch.Tensor:
                cur_img = cur_img.permute(1,2,0).cpu().numpy()

            if type(cur_points) == torch.Tensor:
                cur_points = cur_points.cpu().numpy()

            self.set_image(cur_img)
            colors = 'b'
            if point_labels is not None:
                num_classes = (point_labels[i].max() + 1).cpu().long().item()
                cmap = plt.cm.get_cmap('tab20b', num_classes)
                color_map = np.array([to_rgba(cmap(i))[:3] for i in range(num_classes)])
                random.shuffle(color_map)

                colors = (color_map[point_labels[i].long().cpu().numpy()] * 255).astype(np.uint8).tolist()
                colors = [tuple(x) for x in colors]

            if len(cur_points) > 0:
                self.draw_points(cur_points, colors=colors, sizes=np.array([0.1] * len(cur_points)))
            poly_img = self.get_image()
            img_log[prefix].append(
                self.wandb.Image(poly_img)
            )
        self.wandb.log(img_log)

    def _vis_super_pixel(self, prefix, masks):
        img_log = {}
        img_log[prefix] = []
        num_classes = masks.max() + 1
        # mask_one_hot = F.one_hot(masks.long(), num_classes=num_classes).permute(0,3,1,2)

        cmap = plt.cm.get_cmap('tab20b', num_classes)
        color_map = np.array([to_rgba(cmap(i))[:3] for i in range(num_classes)])
        np.random.shuffle(color_map)

        for i in range(len(masks)):
            cur_img = masks[i].cpu().numpy()
            vis_img = color_map[cur_img]
            vis_img[cur_img == 0] = 0
            # cur_mask = mask_one_hot[i].bool()

            # self.set_image(cur_mask)
            # self.draw_binary_masks(cur_img, colors=color_map, alphas=[0] * num_classes)
            # vis_img = self.get_image()
            img_log[prefix].append(
                self.wandb.Image(vis_img)
            )
        self.wandb.log(img_log)


    def _vis_featmap(self, prefix, imgs, feats):
        img_log = {}
        img_log[prefix] = []
        imgs = self.de_normalize(imgs)

        for i in range(len(imgs)):
            cur_img = imgs[i]
            if type(cur_img) == torch.Tensor:
                cur_img = cur_img.permute(1,2,0).cpu().numpy()

            self.set_image(np.zeros(cur_img.shape[:2]))
            featmap = self.draw_featmap(feats[i], topk=1, resize_shape=cur_img.shape[:2])
            img_log[prefix].append(
                self.wandb.Image(featmap)
            )

        self.wandb.log(img_log)

    def _vis_masks(self, prefix, imgs, masks):
        img_log = {}
        img_log[prefix] = []

        imgs = self.de_normalize(imgs)

        for i in range(len(imgs)):
            cur_img = imgs[i]
            if type(cur_img) == torch.Tensor:
                cur_img = cur_img.permute(1,2,0).cpu().numpy()

            if type(masks) == torch.Tensor:
                masks = masks.cpu().numpy()

            wandb_masks = {}
            wandb_masks['mask1'] = {
                'mask_data': masks[i],
                'class_labels': {0: 'background', 1: 'mask'}
            }

            img_log[prefix].append(
                self.wandb.Image(
                    cur_img,
                    masks = wandb_masks,
                    classes = self.wandb.Classes([{'id':1, 'name': 'mask'}])
                )
            )

        self.wandb.log(img_log)


    def after_train_iter(self, runner):
        # if self.get_mode(runner) == 'train':
        # #     # An ugly patch. The iter-based eval hook will call the
        # #     # `after_train_iter` method of all logger hooks before evaluation.
        # #     # Use this trick to skip that call.
        # #     # Don't call super method at first, it will clear the log_buffer
        #     return super(WandbVisualizer, self).after_train_iter(runner)
        # else:
        super(WandbVisualizer, self).after_train_iter(runner)

        if self.every_n_iters(runner, self.interval):
            results = runner.outputs
            if 'states' in results:
                self.process_vis_data(results['states'])

        if self.every_n_iters(runner, self.scalar_interval):
            results = runner.outputs
            if 'log_vars' in results:
                self.wandb.log(results['log_vars'])


    def _update_wandb_config(self, runner):
        """Update wandb config."""
        # Import the config file.
        sys.path.append(runner.work_dir)
        config_filename = runner.meta['exp_name'][:-3]
        configs = importlib.import_module(config_filename)
        # Prepare a nested dict of config variables.
        config_keys = [key for key in dir(configs) if not key.startswith('__')]
        config_dict = {key: getattr(configs, key) for key in config_keys}
        # Update the W&B config.
        self.wandb.config.update(config_dict)

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image', 'polygon', 'footprint']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'ground_truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)


    def _add_ground_truth(self):

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        CLASSES = self.val_dataset.CLASSES
        self.class_id_to_label = {
            id + 1: name
            for id, name in enumerate(CLASSES)
        }
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            mask_name = img_info.get('maskname', f'mask_{idx}')
            img_height, img_width = img_info['height'], img_info['width']

            image = img_loader(image_name)
            mask_image = img_loader(mask_name)

            # Get image and convert from BGR to RGB
            # image = mmcv.bgr2rgb(img_meta['img'])

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']
            masks = data_ann.get('masks', None)

            # Get dict of bounding boxes to be logged.
            assert len(bboxes) == len(labels)
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels)

            poly_image=None
            if self.vis_mask_as_poly:
                polygons = [np.array(polygon) for polygon in masks]
                polygons = polygons[:self.max_polygon_num]
                # self.set_image(np.zeros(image.size).astype(np.uint8))
                self.set_image(np.array(mask_image))
                self.draw_polygons(polygons, line_widths=1)
                poly_image = self.get_image()
                poly_image = self.wandb.Image(
                    poly_image,
                )

            # Get dict of masks to be logged.
            if masks is not None and not self.without_mask:
                wandb_masks = self._get_wandb_masks(
                    masks,
                    labels,
                    is_poly_mask=True,
                    height=img_height,
                    width=img_width)
            else:
                wandb_masks = None
            # TODO: Panoramic segmentation visualization.

            # Log a row to the data table.
            self.data_table.add_data(
                image_name,
                self.wandb.Image(
                    image,
                    boxes=wandb_boxes,
                    masks=wandb_masks,
                    classes=self.class_set
                ),
                self.wandb.Image(
                    poly_image
                ),
                self.wandb.Image(
                    mask_image
                )
            )

    def _log_predictions(self, results, class_id_to_label=None):
        if class_id_to_label is None:
            class_id_to_label = self.class_id_to_label

        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            # Get the result
            result = results[eval_image_index]
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None

            assert len(bbox_result) == len(class_id_to_label)

            # Get labels
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            # Get segmentation mask if available.
            segms = None
            if segm_result is not None and len(labels) > 0:
                segms = mmcv.concat_list(segm_result)
                segms = mask_util.decode(segms)
                segms = segms.transpose(2, 0, 1)
                assert len(segms) == len(labels)
            # TODO: Panoramic segmentation visualization.

            # Remove bounding boxes and masks with score lower than threshold.
            if type(self.bbox_score_thr) == list or self.bbox_score_thr > 0:
                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > np.array(self.bbox_score_thr)[labels]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]

            # Get dict of bounding boxes to be logged.
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=False,
                                                 class_id_to_label=class_id_to_label)
            # Get dict of masks to be logged.
            if segms is not None:
                wandb_masks = self._get_wandb_masks(segms, labels)
            else:
                wandb_masks = None

            class_set = self.wandb.Classes([{
                'id': id,
                'name': name
            } for id, name in class_id_to_label.items()])
            # Log a row to the eval table.
            self.eval_table.add_data(
                self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.wandb.Image(
                    self.data_table_ref.data[ndx][1],
                    boxes=wandb_boxes,
                    masks=wandb_masks,
                    classes=class_set))

    def _get_wandb_bboxes(self, bboxes, labels, log_gt=True, class_id_to_label=None):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (list): List of bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            labels (int): List of label ids.
            log_gt (bool): Whether to log ground truth or prediction boxes.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        if class_id_to_label is None:
            class_id_to_label = self.class_id_to_label

        wandb_boxes = {}

        box_data = []
        for bbox, label in zip(bboxes, labels):
            if not isinstance(label, int):
                label = int(label)
            label = label + 1

            if len(bbox) == 5:
                confidence = float(bbox[4])
                class_name = class_id_to_label[label]
                box_caption = f'{class_name} {confidence:.2f}'
            else:
                box_caption = str(class_id_to_label[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            box_data.append({
                'position': position,
                'class_id': label,
                'box_caption': box_caption,
                'domain': 'pixel'
            })

        wandb_bbox_dict = {
            'box_data': box_data,
            'class_labels': class_id_to_label
        }

        if log_gt:
            wandb_boxes['ground_truth'] = wandb_bbox_dict
        else:
            wandb_boxes['predictions'] = wandb_bbox_dict

        return wandb_boxes

    def _get_wandb_masks(self,
                         masks,
                         labels,
                         is_poly_mask=False,
                         height=None,
                         width=None):
        """Get list of structured dict for logging masks to W&B.

        Args:
            masks (list): List of masks.
            labels (int): List of label ids.
            is_poly_mask (bool): Whether the mask is polygonal or not.
                This is true for CocoDataset.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            Dictionary of masks to be logged.
        """
        mask_label_dict = dict()
        for mask, label in zip(masks, labels):
            label = label + 1
            # Get bitmap mask from polygon.
            if is_poly_mask:
                if height is not None and width is not None:
                    from rsidet.core.mask.structures import polygon_to_bitmap
                    mask = polygon_to_bitmap(np.array(mask), height, width)
            # Create composite masks for each class.
            if label not in mask_label_dict.keys():
                mask_label_dict[label] = mask
            else:
                mask_label_dict[label] = np.logical_or(mask_label_dict[label],
                                                       mask)

        wandb_masks = dict()
        for key, value in mask_label_dict.items():
            # Create mask for that class.
            value = value.astype(np.uint8)
            value[value > 0] = key

            # Create dict of masks for logging.
            class_name = self.class_id_to_label[key]
            wandb_masks[class_name] = {
                'mask_data': value,
                'class_labels': self.class_id_to_label
            }

        return wandb_masks

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, idx, with_artifact=True, table_key='table'):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        if with_artifact:
            pred_artifact = self.wandb.Artifact(
                f'run_{self.wandb.run.id}_pred', type='evaluation')
            pred_artifact.add(self.eval_table, 'eval_data')
            if self.by_epoch:
                aliases = ['latest', f'epoch_{idx}']
            else:
                aliases = ['latest', f'iter_{idx}']
            self.wandb.run.log_artifact(pred_artifact, aliases=aliases)
        else:
            self.wandb.log({table_key: self.eval_table})
