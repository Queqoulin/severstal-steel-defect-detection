import os
import pandas as pd
from six import string_types

from skimage import io
# from skimage import novice
import argparse

import numpy as np
import pickle


def rle2mask(rle_str, height, width, defect_class):
    rows, cols = height, width
    rle_nums = [int(numstring) for numstring in rle_str.split(' ')]
    rle_pairs = np.array(rle_nums).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = defect_class
    img = img.reshape(cols, rows)
    img = img.T
    return img


# process input data
def process_input(in_folder, in_csv, out):
    train_info = pd.read_csv(in_csv, sep=',')
    train_info[['ImageId', 'ClassId']] = train_info['ImageId_ClassId'].str.split('_', expand=True)
    train_info = train_info.drop(['ImageId_ClassId'], axis=1)
    train_im_path = in_folder

    imname_list = [f for f in os.listdir(train_im_path)
                   if os.path.isfile(os.path.join(train_im_path, f))]

    defect_class_ids = [1, 2, 3, 4]
    defect_class_num = len(defect_class_ids)

    # kekekeke
    res_tensor = {}

    cur_mask = None
    width, height = 0, 0

    # let's iterate through train info df rather than through imname_list:
    # otherwise at every iteration we'd search for a corresponding set of rows
    # which could be extremely time-consuming (?)
    for index, row in train_info.iterrows():
        f = row['ImageId']

        defect_class = index % defect_class_num + 1
        if defect_class == defect_class_ids[0]:
            cur_path = os.path.join(train_im_path, f)
            cur_arr_gray = io.imread(cur_path, as_gray=True)
            height, width = cur_arr_gray.shape
            cur_mask = np.zeros(shape=cur_arr_gray.shape)
            res_tensor[f] = [cur_arr_gray]

        rle_str = row['EncodedPixels']
        if isinstance(rle_str, string_types):
            rle_str = row['EncodedPixels']
            tmp = rle2mask(rle_str, height, width, defect_class=defect_class)

            # just mere matrix addition
            # TODO check whether it's wise
            cur_mask = np.add(cur_mask, tmp)

        if defect_class == defect_class_ids[-1]:
            res_tensor[f].append(cur_mask)

    # write
    with open(out, 'wb') as f_out:
        pickle.dump(res_tensor, f_out)

    return res_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="in_fname", help="path to train_image/ and train.csv", nargs=2)
    parser.add_argument("-o", dest="out_fname", help="pth to the output file")
    args = parser.parse_args()

    col = args.in_fname
    in_folder = col[0]
    in_csv = col[1]
    out = args.out_fname

    train_tensor = process_input(in_folder, in_csv, out)
