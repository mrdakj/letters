import sys
import os
import pathlib
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

from model import Model

mode = 'one_letter'
position = ''
class_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
img_size = 28
batch_size = 256

AUTOTUNE = tf.data.AUTOTUNE

def get_data():
    if mode == 'one_letter':
        train_dirs = [pathlib.Path('dataset/train/one_letter/normal/prepared'), pathlib.Path('dataset/train/one_letter/medium/prepared'), pathlib.Path('dataset/train/one_letter/bold/prepared')]
        val_dirs = [pathlib.Path('dataset/validation/one_letter/normal/prepared'), pathlib.Path('dataset/validation/one_letter/medium/prepared'), pathlib.Path('dataset/validation/one_letter/bold/prepared')]
        test_dirs = [pathlib.Path('dataset/test/one_letter/normal/prepared'), pathlib.Path('dataset/test/one_letter/medium/prepared'), pathlib.Path('dataset/test/one_letter/bold/prepared')]
    else:
        train_dirs = [pathlib.Path(f'dataset/train/two_letters_combined/{position}')]
        val_dirs = [pathlib.Path(f'dataset/validation/two_letters_combined/{position}')]
        test_dirs = [pathlib.Path(f'dataset/test/two_letters_combined/{position}')]

    print("reading training data from:", train_dirs)
    print("reading validation data from:", val_dirs)

    train_ds_list = tf.data.Dataset.list_files(str(train_dirs[0]/'*/*'), shuffle=True)
    for train_dir in train_dirs[1:]:
        train_ds_list = train_ds_list.concatenate(tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=True))

    image_count = len(train_ds_list)
    train_ds_list = train_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    train_ds = train_ds_list.take(image_count)
    # train_ds = train_ds_list.take(100)

    val_ds_list = tf.data.Dataset.list_files(str(val_dirs[0]/'*/*'), shuffle=True)
    for val_dir in val_dirs[1:]:
        val_ds_list = val_ds_list.concatenate(tf.data.Dataset.list_files(str(val_dir/'*/*'), shuffle=True))

    image_count = len(val_ds_list)
    val_ds_list = val_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    val_ds = val_ds_list.take(image_count)
    # val_ds = val_ds_list.take(100)

    test_ds_list = tf.data.Dataset.list_files(str(test_dirs[0]/'*/*'), shuffle=True)
    for test_dir in test_dirs[1:]:
        test_ds_list = test_ds_list.concatenate(tf.data.Dataset.list_files(str(test_dir/'*/*'), shuffle=True))

    image_count = len(test_ds_list)
    test_ds_list = test_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    test_ds = test_ds_list.take(image_count)
    # test_ds = test_ds_list.take(100)

    return train_ds, val_ds, test_ds


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

def analyze_dataset(ds, report_dir, title):
    ds_letters = list(ds.map(lambda file_path: tf.strings.split(file_path, os.path.sep)[-2], num_parallel_calls=AUTOTUNE))
    ds_letters_indexes = list(map(lambda x: ord(x.numpy().decode("utf-8")) - 97, ds_letters))
    bincounts = np.bincount(ds_letters_indexes)
    plt.title(f'Velicina skupa: {np.sum(bincounts)}', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.bar(class_names, bincounts)
    plt.savefig(f'{report_dir}/{title}.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    global mode, position

    if len(sys.argv) != 2:
        sys.exit('Usage: network.py [one, first, second]')

    mode = 'one_letter' if sys.argv[1] == 'one' else 'two_letters'

    if sys.argv[1] == 'first':
        position = 'first'
    elif sys.argv[1] == 'second':
        position = 'second'

    print(mode, position)

    model_dir =  f'model/{mode}' if position == '' else f'model/{mode}/{position}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    report_dir = f'report/{mode}' if position == '' else f'report/{mode}/{position}'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    train_ds, val_ds, test_ds = get_data()
    print('data read')
    analyze_dataset(train_ds, report_dir, 'trening_podaci_raspodela')
    analyze_dataset(val_ds, report_dir, 'validacioni_podaci_raspodela')
    analyze_dataset(test_ds, report_dir, 'test_podaci_raspodela')

    print("Train size", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation size", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test size", tf.data.experimental.cardinality(test_ds).numpy())

    # set num_parallel_calls so multiple images are loaded/processed in parallel
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    model = Model(img_size, class_names, model_dir, report_dir)
    model.train(train_ds, val_ds)
    model.plot_accuracy_loss()

    # load model with best val accuracy
    model.load_best_model()
    model.print_scores(train_ds, val_ds, test_ds)
    model.dump_report(val_ds, test_ds)

if __name__ == "__main__":
    main()

