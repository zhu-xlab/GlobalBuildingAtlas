from .builder import DATASETS
from .custom import EODataset
import pdb
import geopandas as gpd
import tifffile
import rasterio
import numpy as np
from shapely.geometry import Polygon
import os
from pyproj import CRS
from .pipelines import Compose, LoadImageFromFileV3
from rsipoly.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from collections import OrderedDict

@DATASETS.register_module()
class SRTrainingDataset(EODataset):


    def __init__(self, gt_seg_map_loader_pipeline=None, **kwargs):
        super(SRTrainingDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = Compose(gt_seg_map_loader_pipeline) if gt_seg_map_loader_pipeline is not None else LoadImageFromFileV3(imdecode_backend='tifffile')
        # CLASSES = ('building', 'background')
        CLASSES = ('background', 'building')
        PALETTE = [[0, 0, 0], [255, 255, 255]]
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE

    def format_results(self, results, imgfile_prefix, out_dir=None, img_metas=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        assert len(img_metas) == 1
        filename = img_metas[0]['filename']
        city_name = filename.split('/')[-1].split('.')[0]
        crs = img_metas[0]['crs']
        transform = img_metas[0]['geo_transform']

        if 'loss' in results:
            seg_path = img_metas[0]['seg_path']
            loss_out_path = os.path.join(out_dir, 'loss.txt')
            loss = results['loss']
            with open(loss_out_path, 'a+') as f:
                f.write(f'{filename} {seg_path} {loss}\n')

        """
        if 'pred_mask' in results:
            pred_mask = results['pred_mask'][0]
            out_path = os.path.join(out_dir, city_name + '.tif')

            with rasterio.open(
                out_path, 'w',
                driver='GTiff',
                height=pred_mask.shape[0],
                width=pred_mask.shape[1],
                count=1,  # Number of bands
                dtype=pred_mask.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(pred_mask, 1)  # Write array to the first band
        """

        if 'polygons' in results:
            city_transform = self.transforms[city_name]

            start_x, start_y = filename.split('/')[-1].split('.')[0].split('_')[-1].split('x')
            start_x, start_y = int(start_x), int(start_y)
            offset = np.array((start_x, start_y)).reshape(1, 2)

            polygons = results['polygons'][0]
            offset_polygons = [(polygon / self.upscale + offset) for polygon in polygons]

            if not city_name in self.global_polygons.keys():
                self.global_polygons[city_name] = []

            for polygon in offset_polygons:
                new_polygon = []
                for point in polygon:
                    new_point = self.transforms[city_name] * point
                    new_polygon.append(new_point)
                new_polygon = Polygon(new_polygon)
                self.global_polygons[city_name].append(new_polygon)

        return [[]]

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        results = self.img_infos[idx]
        # self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        results = self.img_infos[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_map_by_idx(self, idx):
        """Get one ground truth segmentation map for evaluation."""
        results = self.img_infos[idx]

        # self.pre_pipeline(results)
        results = self.gt_seg_map_loader(results)
        mask = results['gt_semantic_seg']
        return mask
        # return np.expand_dims(mask)

    def pre_eval(self, result, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference

        preds = result[0]['pred_mask']
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        metric = ['mIoU', 'mFscore']

        # cities = [x[1]['city_name'] for x in results]
        if 'continent_name' in results[0][1] and results[0][1]['continent_name'] is not None:
            cities = [x[1]['continent_name'] for x in results]
        else:
            cities = [x[1]['city_name'] for x in results]

        all_results = [x[0] for x in results]

        city_names = np.unique(cities).tolist() + ['all']
        city_iou = {}

        for city_name in city_names:
            idxes = (np.array(cities) == city_name).nonzero()[0]
            results = [all_results[x] for x in idxes] if city_name != 'all' else all_results

            eval_results = {}
            # test a list of files
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                if gt_seg_maps is None:
                    gt_seg_maps = self.get_gt_seg_maps()

                num_classes = len(self.CLASSES)
                ret_metrics = eval_metrics(
                    results,
                    gt_seg_maps,
                    num_classes,
                    self.ignore_index,
                    metric,
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label)
            # test a list of pre_eval_results
            else:
                ret_metrics = pre_eval_to_metrics(results, metric)

            # Because dataset.CLASSES is required for per-eval.
            if self.CLASSES is None:
                class_names = tuple(range(num_classes))
            else:
                class_names = self.CLASSES

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            city_iou[city_name] = ret_metrics['IoU'][1]

            # each class table
            ret_metrics.pop('aAcc', None)
            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)

            # for logger
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                if key == 'aAcc':
                    summary_table_data.add_column(key, [val])
                else:
                    summary_table_data.add_column('m' + key, [val])

            print_log(f'\n per class results for city {city_name}:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            # print_log('Summary:', logger)
            # print_log('\n' + summary_table_data.get_string(), logger=logger)

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'aAcc':
                    eval_results[key] = value / 100.0
                else:
                    eval_results['m' + key] = value / 100.0

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                eval_results.update({
                    key + '.' + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                })

        all_values = []
        for city_name, value in city_iou.items():
            if city_name != 'all':
                print_log(f'\n Building IoU for {city_name} = {value}', logger)
                all_values.append(value)

        print_log(f'\n Averaged building IoU for all cities = {sum(all_values) / len(all_values)}:')

        return eval_results
