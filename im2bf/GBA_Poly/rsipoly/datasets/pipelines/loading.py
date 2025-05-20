import os.path as osp
import pdb
from time import time
import os

import mmcv
import numpy as np
from cv2 import imread
import cv2
import tifffile as tiff
from pycocotools import mask as cocomask

from ..builder import PIPELINES
import rasterio
import shapely
import geojson
import warnings

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 use_mask_as_img=True,
                 combine_mask_with_img=False,
                 binarize_mask=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.use_mask_as_img = use_mask_as_img
        self.combine_mask_with_img = combine_mask_with_img
        self.binarize_mask = binarize_mask

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # if results.get('img_prefix') is not None:
        #     filename = osp.join(results['img_prefix'],
        #                         results['img_info']['filename'])
        # else:
        filename = results['img_info']['filename']
        maskname = results['img_info']['maskname']

        img_bytes = self.file_client.get(filename)
        mask_bytes = self.file_client.get(maskname)

        if self.use_mask_as_img:
            img = mmcv.imfrombytes(
                mask_bytes, flag=self.color_type, backend=self.imdecode_backend)

        elif self.combine_mask_with_img:
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            mask = mmcv.imfrombytes(
                mask_bytes, flag=self.color_type, backend=self.imdecode_backend)
            img = np.concatenate([img, mask[:, :, 0:1]], axis=-1)

            if self.binarize_mask:
                mask = (mask > 0).astype(np.uint8)

            results['gt_semantic_seg'] = mask[:, :, 0:1]
            results['seg_fields'] = ['gt_semantic_seg']
        else:
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            mask = mmcv.imfrombytes(
                mask_bytes, flag=self.color_type, backend=self.imdecode_backend)
            results['gt_semantic_seg'] = mask[:,:,0]
            results['seg_fields'] = ['gt_semantic_seg']

        # img = np.concatenate([img, mask], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = 'Horizontal'


        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadRasterFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(
        self, to_float32=False, color_type='color',
        file_client_args=dict(backend='disk'),
        imdecode_backend='cv2', binarize=True
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.binarize = binarize

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['img_info']['filename']
        maskname = results['img_info'].get('maskname', None)

        temp = rasterio.open(filename)
        transform = temp.transform
        CRS = temp.crs
        img = temp.read()
        img = np.transpose(img, [1,2,0]).repeat(3,2)
        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(
        #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)


        mask_bytes = self.file_client.get(maskname) if maskname is not None else None


        if self.binarize:
            img = (img > 0).astype(np.uint8)

        if self.to_float32:
            img = img.astype(np.float32)

        if mask_bytes:
            mask = mmcv.imfrombytes(
                mask_bytes, flag=self.color_type, backend=self.imdecode_backend)
            results['gt_semantic_seg'] = mask[:, :, 0:1]
            results['seg_fields'] = ['gt_semantic_seg']


        results['filename'] = filename
        results['ori_filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = 'Horizontal'
        results['geo_transform'] = transform
        results['geo_crs'] = CRS

        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False
        )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotationsGTA(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        #img_bytes = self.file_client.get(filename)
        gt_semantic_seg = imread(filename, 2) / 100.
        #gt_semantic_seg = imread(filename, 2)
        gt_semantic_seg = np.clip(gt_semantic_seg, 0, 500)
        if np.isnan(gt_semantic_seg.sum()):
            gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
        # modify if custom classes
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class LoadAnnotationsDepth(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        #img_bytes = self.file_client.get(filename)
        
        #filename = filename[:-7]+'.png'
        filename = filename.replace('RGB','AGL')
        
        gt_semantic_seg = imread(filename, 2)
        #gt_semantic_seg = imread(filename, 2) / 100.
        gt_semantic_seg[gt_semantic_seg>400] = 0
        #gt_semantic_seg = mmcv.imread(filename,2)
        gt_semantic_seg = np.clip(gt_semantic_seg, 0, 400)
        # If these is NaN value
        #if np.isnan(gt_semantic_seg.sum()):
        #    gt_semantic_seg = np.where(np.isnan(gt_semantic_seg), np.full_like(gt_semantic_seg, 0), gt_semantic_seg)
        '''gt_semantic_seg = mmcv.imfrombytes(
            iimg_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()'''
        # modify if custom classes
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFile_MS(object):
    """Load a multispectral tif image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        #img_bytes = self.file_client.get(filename)
        #img = mmcv.imfrombytes(
        #    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        with rasterio.open(filename,'r') as rf:
            img = rf.read() # (C,W,H)
            img = np.transpose(img,(1,2,0))
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadShapeFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class LoadRasterWithWindow(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(
        self, binarize=True, repeat=False
    ):
        self.binarize = binarize
        self.repeat=repeat

    def get_patch_geotransform(self, src, window):
        """
        Calculate the geotransformation of a patch from a rasterio dataset.

        Parameters:
        - src: rasterio dataset
        - window: rasterio window object defining the patch
        
        Returns:
        - Geotransformation tuple for the patch.
        """
        # Get the original geotransformation
        original_transform = src.transform
        
        # Calculate the new top left corner's geographic coordinates
        # The offset is given by window.col_off and window.row_off in pixels
        # Adjust the origin using the pixel sizes (src.res[0] for x, src.res[1] for y)
        new_top_left_x = original_transform.c + window.col_off * original_transform.a
        new_top_left_y = original_transform.f + window.row_off * original_transform.e
        
        # Create a new geotransformation for the patch
        patch_transform = rasterio.Affine(original_transform.a, original_transform.b, new_top_left_x,
                                          original_transform.d, original_transform.e, new_top_left_y)
        
        return patch_transform


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = str(results['img_info']['filename'])
        maskname = results['img_info'].get('maskname', None)

        crop_boxes = results['img_info']['crop_boxes']

        # Avoid loading the file if the target file already exists
        in_root = results['in_root']
        out_root = results['out_root']

        rel_path = os.path.relpath(filename, in_root)
        out_path = os.path.join(out_root, rel_path)
        out_path = out_path.split('.tif')[0]

        crop_boxes_str = [str(x) for x in crop_boxes]
        out_path = os.path.join(out_path, '_'.join(crop_boxes_str))

        results['out_path'] = out_path
        results['filename'] = filename
        results['crop_boxes'] = crop_boxes
        results['ori_filename'] = filename

        if os.path.exists(out_path):
            # img = np.zeros((crop_boxes[2], crop_boxes[3], 4))
            img = None
            transform = None
            CRS = None

        else:
            try:
                os.makedirs(out_path)
                window = rasterio.windows.Window(col_off=crop_boxes[0], row_off=crop_boxes[1],
                                                 width=crop_boxes[3], height=crop_boxes[2])
                with rasterio.open(filename) as src:
                    # transform = src.transform
                    transform = self.get_patch_geotransform(src, window)
                    CRS = src.crs
                    img = src.read(window=window)
                    img = np.transpose(img, [1,2,0])
                    if self.repeat or img.shape[-1] == 1:
                        img = img.repeat(3,2)

            except Exception as e:
                img = np.zeros((crop_boxes[2], crop_boxes[3], 4))
                transform = None
                CRS = None
                print(e)

            if self.binarize:
                img = (img > 0).astype(np.uint8)

            results['img'] = img
            results['scale_factor'] = 1.0
            results['flip'] = False
            results['flip_direction'] = 'Horizontal'
            results['geo_transform'] = transform
            results['geo_crs'] = CRS

            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['pad_shape'] = img.shape

            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False
            )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFileV2(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 binarize_mask=True
            ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.binarize_mask = binarize_mask

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if 'maskname' in results['img_info']:
            maskname = results['img_info']['maskname']
            mask_bytes = self.file_client.get(maskname)
            mask = mmcv.imfrombytes(mask_bytes, flag='grayscale', backend='cv2')
            np.expand_dims(mask, axis=2)
            if self.binarize_mask:
                mask = (mask > 0).astype(np.uint8)
            results['seg_fields'] = ['gt_semantic_seg']
            results['gt_semantic_seg'] = mask


        # results['gt_semantic_seg'] = mask[:,:,0]
        # results['seg_fields'] = ['gt_semantic_seg']

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = 'Horizontal'


        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadFeatures(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self):
        pass

    def __call__(self, results):

        features = results['ann_info']['features']
        raster_shape = results['img_shape'][:-1]


        shapes = []
        new_features = []
        for i, feature in enumerate(features):
            feature = [np.array(x).reshape(-1,2).tolist() for x in feature]
            exterior = feature[0]
            interiors = [] if len(feature) == 1 else feature[1:]
            polygon = geojson.Polygon([exterior], interiors)
            new_features.append(polygon)
            shapes.append((polygon, i + 1))

        if len(shapes) > 0:
            raster = rasterio.features.rasterize(
                shapes, out_shape=raster_shape, dtype=np.int32, all_touched=True
            )
        else:
            raster = np.zeros(raster_shape, dtype=np.int32)

        results['gt_semantic_seg'] = raster
        results['features'] = new_features
        results['seg_fields'] = ['gt_semantic_seg']

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadFeaturesV2(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self):
        pass

    def __call__(self, results):

        geojson_path = results['ann_info']['features']
        raster_shape = results['img_shape'][:-1]

        with open(geojson_path, 'r') as f:
            features = geojson.load(f)


        shapes = []
        new_features = []
        for i, feature in enumerate(features):
            feature = [np.array(x).reshape(-1,2).tolist() for x in feature['coordinates']]
            new_features.append(feature)
            exterior = feature[0]
            interiors = [] if len(feature) == 1 else feature[1:]
            polygon = geojson.Polygon([exterior], interiors)
            shapes.append((polygon, i + 1))

        if len(shapes) > 0:
            raster = rasterio.features.rasterize(
                shapes, out_shape=raster_shape, dtype=np.int32, all_touched=True
            )
        else:
            raster = np.zeros(raster_shape, dtype=np.int32)

        results['gt_semantic_seg'] = raster
        results['features'] = new_features
        results['seg_fields'] = ['gt_semantic_seg']

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFileV3(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 binarize_mask=True,
                 collect_transform=False,
                 use_shp=False,
                 raster_shape=(1024, 1024),
                 raster_downscale=1.,
                 reverse_gt=False,
                 collect_features=False,
                 raster_offsets=None,
            ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.binarize_mask = binarize_mask
        self.collect_transform = collect_transform
        self.use_shp = use_shp
        self.raster_shape = raster_shape
        self.raster_downscale = raster_downscale
        self.reverse_gt = reverse_gt
        self.collect_features = collect_features
        self.raster_offsets = raster_offsets

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        new_results = {}

        if 'img_path' in results:
            filename = results['img_path']

            warnings.filterwarnings("ignore")
            raster = rasterio.open(filename)
            img = raster.read()
            img = np.transpose(img, [1,2,0])
            if self.collect_transform:
                new_results['geo_transform'] = raster.transform
                new_results['crs'] = raster.crs

            # img_bytes = self.file_client.get(filename)
            # img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            # img = img.astype(np.float32)
            # img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend='cv2')
            # img = np.zeros((256, 256, 4), dtype=np.uint8)
            # results['filename'] = filename

        if 'seg_path' in results:
            maskname = results['seg_path']
            if maskname is None:
                H, W, _ = img.shape
                mask = np.zeros((H, W))
            elif maskname.endswith('.tif'):
                mask_bytes = self.file_client.get(maskname)
                mask = mmcv.imfrombytes(mask_bytes, flag='grayscale', backend='tifffile')
            else:
                mask_bytes = self.file_client.get(maskname)
                mask = mmcv.imfrombytes(mask_bytes, flag='grayscale', backend='cv2')

            np.expand_dims(mask, axis=2)
            if self.binarize_mask:
                mask = (mask > 0).astype(np.uint8)
                if self.reverse_gt:
                    mask = 1 - mask

            new_results['seg_fields'] = ['gt_semantic_seg']
            new_results['gt_semantic_seg'] = mask
            new_results['seg_path'] = results['seg_path']

        if 'ann_path' in results and self.use_shp:
            ann_path = results['ann_path']
            H, W, _ = img.shape
            try:
                jsons = geojson.load(open(ann_path, 'r'))
            except:
                jsons = []
            raster_gt = self.rasterize([jsons], downscale=self.raster_downscale, raster_shape=self.raster_shape, all_touched=False)[0]
            new_results['gt_semantic_seg'] = raster_gt


        if 'features' in results and self.use_shp:
            features = results['features']
            raster_gt = self.rasterize([features], downscale=self.raster_downscale, raster_shape=self.raster_shape, all_touched=False)[0]
            if self.binarize_mask:
                raster_gt = (raster_gt > 0).astype(np.uint8)

            new_results['seg_fields'] = ['gt_semantic_seg']
            new_results['gt_semantic_seg'] = raster_gt

        if 'ndsm_path' in results:
            filename = results['ndsm_path']
            if filename is not None:
                ndsm_bytes = self.file_client.get(filename)
                ndsm = mmcv.imfrombytes(ndsm_bytes, flag=self.color_type, backend=self.imdecode_backend)
                img = np.concatenate([img, np.expand_dims(ndsm, 2)], axis=2)
            else:
                img = np.concatenate([img, np.zeros((H, W, 1))], axis=2)

            new_results['use_ndsm'] = True

        if 'majority_voting_path' in results:
            filename = results['majority_voting_path']
            if filename is None:
                H, W, _ = img.shape
                cur_img = np.zeros((H, W))
            else:
                # cur_bytes = self.file_client.get(filename)
                # cur_img = mmcv.imfrombytes(cur_bytes, flag=self.color_type, backend=self.imdecode_backend)
                # cur_img = cv2.resize(cur_img, img.shape[:2], interpolation=cv2.INTER_NEAREST)
                # cur_img = (cur_img > 0).astype(np.uint8)
                cur_img = rasterio.open(filename).read()
                cur_img = np.transpose(cur_img, [1,2,0])
                cur_img = cv2.resize(cur_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            new_results['majority_voting'] = cur_img
            new_results['seg_fields'].append('majority_voting')

        if 'city_name' in results:
            new_results['city_name'] = results['city_name']
        if 'continent_name' in results:
            new_results['continent_name'] = results['continent_name']

        if 'img_id' in results:
            new_results['img_id'] = results['img_id']

        if self.collect_features:
            if 'ann_path' in results:
                ann_path = results['ann_path']
                try:
                    jsons = geojson.load(open(ann_path, 'r'))
                except Exception:
                    jsons = []
                new_results['features'] = jsons
            else:
                assert 'features' in results
                new_results['features'] = results['features']


        if self.to_float32:
            img = img.astype(np.float32)


        new_results['img'] = img
        new_results['filename'] = results['img_path']
        new_results['ori_filename'] = results['img_path']
        new_results['ann_path'] = results['ann_path'] if 'ann_path' in results else None
        new_results['img_shape'] = img.shape
        new_results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        new_results['pad_shape'] = img.shape
        new_results['scale_factor'] = 1.0

        return new_results


        # num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        # results['img_norm_cfg'] = dict(
        #     mean=np.zeros(num_channels, dtype=np.float32),
        #     std=np.ones(num_channels, dtype=np.float32),
        #     to_rgb=False)

        return results

    def rasterize(self, batch_features, downscale=1, raster_shape=(256, 256), add_noise=False,
                  all_touched=True):

        rasters = []
        for features in batch_features:
            shapes = []
            cnt = 1
            for feat in features:
                new_rings = []
                for ring in feat['coordinates']:
                    # exterior = (np.array(feat['exterior']) / downscale - offset).tolist()
                    # interiors = [(np.array(x) / downscale - offset).tolist() for x in feat['interiors']]
                    norm_ring = (np.array(ring) / downscale).tolist()
                    new_rings.append(norm_ring)

                exterior = new_rings[0]
                interiors = [] if len(new_rings) == 1 else new_rings[1:]
                polygon = geojson.Polygon([exterior], interiors)
                shapes.append((polygon, cnt))

                cnt += 1

            if len(shapes) > 0:
                raster = rasterio.features.rasterize(
                    shapes, out_shape=raster_shape, dtype=np.int32, all_touched=True
                )
            else:
                raster = np.zeros(raster_shape, dtype=np.int32)

            rasters.append(raster)

        return np.stack(rasters)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


