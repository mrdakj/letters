import sys
import os
import pathlib
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

from model import Model

class_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
img_size = 28
batch_size = 256

AUTOTUNE = tf.data.AUTOTUNE

def get_data():
    return tf.data.Dataset.list_files(str('dcgan/out/*/*'), shuffle=True)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    return one_hot


def decode_img(img):
    # convert the compressed string to a 1D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    img = img / 255
    # resize the image to the desired size - this doesn't work as cv resize!!!
    # return tf.image.resize(img, [img_size, img_size])
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def main():
    report_dir = f'dcgan/report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    test_ds = get_data()
    print("Test size", tf.data.experimental.cardinality(test_ds).numpy())

    # set num_parallel_calls so multiple images are loaded/processed in parallel
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    test_ds = configure_for_performance(test_ds)

    model = Model('one_letter', img_size, class_names, 'model/one_letter', report_dir)

    # load model with best val accuracy
    model.load_best_model()
    model.print_scores(test_ds)
    model.dump_report(test_ds)

if __name__ == "__main__":
    main()

