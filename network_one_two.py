import os
import pathlib
import numpy as np

import tensorflow as tf

from model import Model

model_dir =  'model/one_two'
report_dir = 'report/one_two'
class_names = np.array(['one', 'two'])
img_size = 28
batch_size = 256

AUTOTUNE = tf.data.AUTOTUNE

def get_data():
    train_dirs = [pathlib.Path('dataset/train/one_letter/normal/prepared'), pathlib.Path('dataset/train/one_letter/medium/prepared'), pathlib.Path('dataset/train/one_letter/bold/prepared'), pathlib.Path('dataset/train/two_letters_combined/first')]

    train_ds_list = tf.data.Dataset.list_files(str(train_dirs[0]/'*/*'), shuffle=True)
    for train_dir in train_dirs[1:]:
        train_ds_list = train_ds_list.concatenate(tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=True))

    image_count = len(train_ds_list)
    train_ds_list = train_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    train_ds = train_ds_list.take(image_count // 2)
    # train_ds = train_ds_list.take(100)

    val_dirs = [pathlib.Path('dataset/validation/one_letter/normal/prepared'), pathlib.Path('dataset/validation/one_letter/medium/prepared'), pathlib.Path('dataset/validation/one_letter/bold/prepared'), pathlib.Path('dataset/validation/two_letters_combined/first')]
    val_ds_list = tf.data.Dataset.list_files(str(val_dirs[0]/'*/*'), shuffle=True)
    for val_dir in val_dirs[1:]:
        val_ds_list = val_ds_list.concatenate(tf.data.Dataset.list_files(str(val_dir/'*/*'), shuffle=True))

    image_count = len(val_ds_list)
    val_ds_list = val_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    val_ds = val_ds_list.take(image_count)
    # val_ds = val_ds_list.take(100)

    test_dirs = [pathlib.Path('dataset/test/one_letter/normal/prepared'), pathlib.Path('dataset/test/one_letter/medium/prepared'), pathlib.Path('dataset/test/one_letter/bold/prepared'), pathlib.Path('dataset/test/two_letters_combined/first')]
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
    one_hot = parts[-3] == ['prepared', 'first']
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
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    train_ds, val_ds, test_ds = get_data()
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

