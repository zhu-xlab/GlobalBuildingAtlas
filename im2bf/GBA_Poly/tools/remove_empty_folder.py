import os
import pdb

# path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/AFRICA'
# path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/ASIAEAST'
# path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/ASIAWEST'
# path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/EUROPE'
# path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/ISLANDS'
# pattern = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/{}'
# paths = [pattern.format(x) for x in ['AFRICA', 'ASIAEAST', 'ASIAWEST', 'EUROPE', 'ISLANDS', 'NORDAMERICA', 'OCEANIA', 'SOUTHAMERICA']]

# pattern = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/{}'
pattern = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/mask/{}/mosaic'
paths = [pattern.format(x) for x in ['africa', 'asiaeast', 'asiawest', 'europe', 'nordamerica', 'oceania', 'southamerica']]

for path in paths:
    """
    Recursively removes empty sub-folders from the specified path.
    """
    # Check whether the specified path is an existing directory
    if not os.path.isdir(path):
        print(f"Error: {path} is not a directory or does not exist.")

    # Walk through the directory tree, from bottom up, so we can safely remove empty sub-directories.
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")

