from .builder import DATASETS
from .custom import EODataset
import pdb
import geopandas as gpd
import tifffile
import rasterio
import numpy as np
import shapely
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
import rsipoly.utils.polygon_utils_lydorn as polygon_utils
from rsipoly.datasets.coco_utils import COCOeval

from pycocotools.coco import COCO
import io
import json
import tempfile




@DATASETS.register_module()
class CrowdAIDataset(EODataset):

    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, coco_ann_path=None, gt_seg_map_loader_pipeline=None, **kwargs):
        super(CrowdAIDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = Compose(gt_seg_map_loader_pipeline) if gt_seg_map_loader_pipeline is not None else LoadImageFromFileV3(imdecode_backend='tifffile')
        self.coco_ann_path = coco_ann_path

    def format_results(self, results, imgfile_prefix, out_dir=None, img_metas=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        assert len(img_metas) == 1
        filename = img_metas[0]['filename']
        city_name = filename.split('/')[-1].split('.')[0]
        transform = img_metas[0]['geo_transform']
        crs = img_metas[0]['crs']

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
        return results
        mask = results['gt_semantic_seg']
        return mask
        # return np.expand_dims(mask)

    def pre_eval(self, results, indices):
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

        pre_eval_results = []
        for result, index in zip(results, indices):
            gt_dict = self.get_gt_seg_map_by_idx(index)
            cur_eval_result = {}

            if 'polygons' in result:
                dt_polygons = result['polygons']
                gt_features = gt_dict['features']
                gt_polygons = [shapely.geometry.shape(gt_feature) for gt_feature in gt_features]
                cur_eval_result['polygons'] = dt_polygons

                if False:
                    fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons, buffer=0.0001)
                    fixed_dt_polygons = polygon_utils.fix_polygons(dt_polygons)
                    mtas = polygon_utils.compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)
                    cur_eval_result['mta'] = mtas

            if 'pred_mask' in result:
                pred = result['pred_mask']
                # if not isinstance(indices, list):
                #     indices = [indices]
                # if not isinstance(preds, list):
                #     preds = [preds]

                seg_map = gt_dict['gt_semantic_seg']
                pre_iou = intersect_and_union(
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
                        reduce_zero_label=self.reduce_zero_label
                )
                cur_eval_result['pre_iou'] = pre_iou

            if 'scores' in result:
                cur_eval_result['scores'] = result['scores']


            pre_eval_results.append(cur_eval_result)

        return pre_eval_results


    def polygons2annotations(self, polygons, img_id, ann_start_id, scores=None):
        annotations = []
        bounds = []
        for i, poly in enumerate(polygons):
            # Flatten the polygon coordinates for COCO
            if type(poly) == dict:
                json_dict = poly
                coords = json_dict['coordinates']
                poly = Polygon(shell=coords[0], holes=coords[1:])
            else:
                json_dict = shapely.geometry.mapping(poly)

            coords = json_dict['coordinates']
            segmentation = [np.array(x).reshape(-1).tolist() for x in coords]

            if json_dict['type'] != 'Polygon':
                pdb.set_trace()

            # exterior = list(poly.exterior.coords)
            # segmentation = [coord for point in exterior for coord in point]
            bounds.append(poly.bounds)
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            bbox_coco_format = [minx + width/2, miny + height/2, width, height]

            annotation = {
                "id": i + ann_start_id,
                "image_id": img_id,  # Assuming all polygons belong to one image
                "category_id": 100,  # Assuming all polygons belong to one category
                "segmentation": segmentation,
                "area": poly.area,  # This is approximate and may need more accurate calculation
                "bbox": bbox_coco_format,  # [minx, miny, width, height]
                "iscrowd": 0,
                "score": scores[i] if scores is not None else 1
            }
            annotations.append(annotation)

        return annotations, np.array(bounds)

    def get_coco_json(self, img_metas, polygons, scores=None):
        ann_start_id = 1
        ann_infos = []
        img_infos = []
        categories = [
            {"id": 100, "name": "building"},
        ]
        bounds = {}
        for i, (img_meta, cur_polygons) in enumerate(zip(img_metas, polygons)):
            img_id = img_meta['img_id']
            filename = img_meta['filename']
            cur_anns, cur_bounds = self.polygons2annotations(cur_polygons, img_id, ann_start_id, scores=scores[i])
            bounds[img_id] = cur_bounds
            ann_start_id += len(cur_anns)
            ann_infos.extend(cur_anns)
            img_infos.append({
                "id": img_id,
                "width": img_meta['ori_shape'][1],
                "height": img_meta['ori_shape'][0],
                "file_name": filename
            })

        coco_format = {
            "images": img_infos,
            "annotations": ann_infos,
            "categories": categories
        }
        coco_str = json.dumps(coco_format)
        return coco_str, bounds

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

        eval_results = {}

        if 'polygons' in results[0][0]:

            dt_polygons = [x[0]['polygons'] for x in results]
            img_metas = [x[1] for x in results]
            scores = [x[0].get('scores', None) for x in results]

            coco_dt_str, dt_bounds = self.get_coco_json(img_metas, dt_polygons, scores=scores)

            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                temp_file.write(coco_dt_str)
                temp_file.flush()  # Ensure all data is written to the file

                coco_dt = COCO(temp_file.name)

            # if not os.path.exists(self.coco_ann_path):

            assert self.coco_ann_path is not None
            """
            gt_polygons = [json.load(open(img_meta['ann_path'])) for img_meta in img_metas]
            coco_gt_str, gt_bounds = self.get_coco_json(img_metas, gt_polygons)
            with open(self.coco_ann_path, 'w') as f:
                f.write(coco_gt_str)
            """
            coco_gt = COCO(self.coco_ann_path)
            polygon_utils.compute_IoU_cIoU(coco_dt, coco_gt)

            coco_eval = COCOeval(coco_gt, coco_dt, DtImgIds=coco_dt.getImgIds(),
                                 gt_bounds=None, dt_bounds=dt_bounds)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            coco_result = coco_eval.result_str

            eval_results['COCO Results'] = coco_result

        if 'mta' in results[0][0]:
            mtas = []
            for result in results:
                mtas.extend(result[0]['mta'])
            mtas = np.array([mta for mta in mtas if mta is not None])

            print_log(f'\n mtas: {mtas.mean()} \n')
            eval_results['Max Tangent Angle Error'] = mtas.mean()

        if 'pre_iou' in results[0][0]:
            metric = ['mIoU', 'mFscore']
            pre_ious = [x[0]['pre_iou'] for x in results]
            # city_names = np.unique(cities).tolist() + ['all']
            city_iou = {}
            iou_eval_result = {}
            # idxes = (np.array(cities) == city_name).nonzero()[0]
            # cur_pre_ious = [pre_ious[x] for x in idxes] if city_name != 'all' else pre_ious

            # ret_metrics = pre_eval_to_metrics(cur_pre_ious, metric)
            ret_metrics = pre_eval_to_metrics(pre_ious, metric)

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

            print_log(f'\n per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            # print_log('Summary:', logger)
            # print_log('\n' + summary_table_data.get_string(), logger=logger)

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'aAcc':
                    iou_eval_result[key] = value / 100.0
                else:
                    iou_eval_result['m' + key] = value / 100.0

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                iou_eval_result.update({
                    key + '.' + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                })

            eval_results['IoU Related'] = iou_eval_result

        return eval_results
