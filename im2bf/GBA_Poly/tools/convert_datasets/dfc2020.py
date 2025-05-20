import os
import numpy as np
import rasterio
from tqdm import tqdm
import argparse
from PIL import Image
import shutil

IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])
DFC2NEW = np.array([0, 1, 2, 0, 3, 4, 5, 6, 0, 7, 8])

def rename_string(s,target):
    # Find the position of the '_lc' substring
    lc_index = s.find(target)
    # Find the position of the '.tif' substring
    tif_index = s.find('.tif')
    # If both substrings are found, construct the new string
    if lc_index != -1 and tif_index != -1:
        # Extract the parts of the string before and after the substrings
        before_lc = s[:lc_index]
        between_lc_tif = s[lc_index+3:tif_index]
        after_tif = s[tif_index:]
        # Construct the new string with the substrings swapped
        new_s = f"{before_lc}{between_lc_tif}{target}{after_tif}"
        return new_s
    else:
        # Return the original string if the substrings are not found
        return "Error: Input string does not contain target sub-string"


parser = argparse.ArgumentParser(
        description='Convert DFC2020 to mmsegmentation format')
parser.add_argument('--in_dir', help='data path')
parser.add_argument('--out_dir', help='output path')
args = parser.parse_args()

#args.in_dir = './'
#args.out_dir = './dfc2020_mmseg'

#os.makedirs(args.out_dir)
s1_dir = os.path.join(args.out_dir,'s1_dir')
s2_dir = os.path.join(args.out_dir,'s2_dir')
ann_dir = os.path.join(args.out_dir,'ann_dir')
os.makedirs(s1_dir+'/train',exist_ok=True)
os.makedirs(s1_dir+'/test',exist_ok=True)
os.makedirs(s2_dir+'/train',exist_ok=True)
os.makedirs(s2_dir+'/test',exist_ok=True)
os.makedirs(ann_dir+'/train',exist_ok=True)
os.makedirs(ann_dir+'/test',exist_ok=True)


## validation
lc_fnames = os.listdir(os.path.join(args.in_dir,'lc_validation/'))
for lc_fname in tqdm(lc_fnames):
        # lc
        with rasterio.open(os.path.join(args.in_dir,'lc_validation/',lc_fname),'r') as rf:
                lc = rf.read(1)
                lc = IGBP2DFC[lc]
                lc = DFC2NEW[lc]
        lc_img = Image.fromarray(lc.astype('uint8'), mode="L")
        lc_fname_new = rename_string(lc_fname,'_lc')
        lc_img.save(ann_dir+'/test/'+lc_fname_new)
        # s2
        s2_fname = lc_fname.replace('lc','s2')
        s2_fname_new = rename_string(s2_fname,'_s2')
        shutil.copy(os.path.join(args.in_dir,'s2_validation/',s2_fname),s2_dir+'/test/'+s2_fname_new)
        '''
        # s1
        s1_fname = lc_fname.replace('lc','s1')
        s1_fname_new = rename_string(s1_fname,'_s1')
        shutil.copy2(s1_fname_new,s1_dir+'/test')
        '''

## validation
lc_fnames = os.listdir(os.path.join(args.in_dir,'lc_0/'))
for lc_fname in tqdm(lc_fnames):
        # lc
        with rasterio.open(os.path.join(args.in_dir,'lc_0/',lc_fname),'r') as rf:
                lc = rf.read(1)
                lc = IGBP2DFC[lc]
                lc = DFC2NEW[lc]
        lc_img = Image.fromarray(lc.astype('uint8'), mode="L")
        lc_fname_new = rename_string(lc_fname,'_lc')
        lc_img.save(ann_dir+'/train/'+lc_fname_new)
        # s2
        s2_fname = lc_fname.replace('lc','s2')
        s2_fname_new = rename_string(s2_fname,'_s2')
        shutil.copy(os.path.join(args.in_dir,'s2_0/',s2_fname),s2_dir+'/train/'+s2_fname_new)
        '''
        # s1
        s1_fname = lc_fname.replace('lc','s1')
        s1_fname_new = rename_string(s1_fname,'_s1')
        shutil.copy2(s1_fname_new,s1_dir+'/test')
        '''
print('Convert finished.')