## Prepare datasets

Datasets are prepared as datapipes from `Dataset4EO`.

### ISPRS Vaihingen

(This is done in `Dataset4EO`)

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip' are required.

For Vaihingen dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size` =512, `stride_size`=256), it will generate 344 images for training and 398 images for validation.
