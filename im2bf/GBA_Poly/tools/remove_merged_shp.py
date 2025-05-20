import os

# path = '/home/Datasets/Dataset4EO/GlobalBF/ai4eo'
path = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs'
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

