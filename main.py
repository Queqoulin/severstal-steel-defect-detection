import os
import pandas as pd
from six import string_types

from skimage import io
from tqdm import tqdm
import argparse

import numpy as np


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


def process_input(in_folder, in_csv):
    train_info = pd.read_csv(in_csv, sep=',')
    train_info[['ImageId', 'ClassId']] = train_info['ImageId_ClassId'].str.split('_', expand=True)
    train_info = train_info.drop(['ImageId_ClassId'], axis=1)
    train_im_path = in_folder

    defect_class_ids = [1, 2, 3, 4]
    defect_class_num = len(defect_class_ids)

    cur_mask = None

    num_examples = train_info.ImageId.unique().shape[0]
    cur_path = os.path.join(train_im_path, train_info.ImageId.values[0])
    cur_arr_gray = io.imread(cur_path, as_gray=True)
    height, width = cur_arr_gray.shape

    x_arr = np.zeros((num_examples, height, width), dtype=np.uint8)
    y_arr = np.zeros((num_examples, height, width), dtype=np.uint8)

    counter = 0
    for index, row in tqdm(train_info.iterrows(), total=train_info.shape[0]):
        f = row['ImageId']

        defect_class = index % defect_class_num + 1
        if defect_class == defect_class_ids[0]:
            cur_path = os.path.join(train_im_path, f)
            cur_arr_gray = io.imread(cur_path, as_gray=True) * 255
            cur_mask = np.zeros(shape=cur_arr_gray.shape, dtype=np.uint8)
            x_arr[counter, :, :] = cur_arr_gray

        rle_str = row['EncodedPixels']
        if isinstance(rle_str, string_types):
            rle_str = row['EncodedPixels']
            tmp = rle2mask(rle_str, height, width, defect_class=defect_class)

            cur_mask = np.add(cur_mask, tmp)

        if defect_class == defect_class_ids[-1]:
            y_arr[counter] = cur_mask
            counter += 1

    assert x_arr.shape[0] == y_arr.shape[0], "Number of targets != number of examples"
    assert len(x_arr.shape) > 1, "It looks like the images are of different shapes"

    return x_arr, y_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="in_fname", help="path to train_image/ and train.csv", nargs=2)
    parser.add_argument("-o", dest="out_fname", help="path to the output file")
    args = parser.parse_args()

    col = args.in_fname
    in_folder = col[0]
    in_csv = col[1]
    out = args.out_fname

    X, y = process_input(in_folder, in_csv)
    np.savez_compressed(out, X=X, y=y)
