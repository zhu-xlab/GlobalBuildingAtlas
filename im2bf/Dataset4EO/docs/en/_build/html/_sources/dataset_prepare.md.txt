## Customize datasets

In this section we briefly describe the procedures to customize a new dataset.

**Step 1.** Create the main datapipe class `YOUR_DATASET.py` in `Dataset4EO/datasets/_builtin/`. If it is a supervised dataset, add class information in `Dataset4EO/datasets/_builtin/YOUR_DATASET.categories`.

The dataset can be either automatically or manually downloaded and organized, depending on your preference.

**Step 2.** Add `from .vaihingen import Vaihingen` to `_builtin/__init__.py`. 

**Step 3.** Create `Dataset4EO/tests/test_YOUR_DATASET.py` to check the datapipe.

For a detailed example see `Tutorials`.