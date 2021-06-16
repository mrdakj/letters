import os
import sys
import pathlib
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

from sklearn import metrics

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

    val_dirs = [pathlib.Path('dataset/validation/one_letter/normal/prepared'), pathlib.Path('dataset/validation/one_letter/medium/prepared'), pathlib.Path('dataset/validation/one_letter/bold/prepared'), pathlib.Path('dataset/validation/two_letters_combined/first')]
    val_ds_list = tf.data.Dataset.list_files(str(val_dirs[0]/'*/*'), shuffle=True)
    for val_dir in val_dirs[1:]:
        val_ds_list = val_ds_list.concatenate(tf.data.Dataset.list_files(str(val_dir/'*/*'), shuffle=True))

    image_count = len(val_ds_list)
    val_ds_list = val_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    val_ds = val_ds_list.take(image_count)

    test_dirs = [pathlib.Path('dataset/test/one_letter/normal/prepared'), pathlib.Path('dataset/test/one_letter/medium/prepared'), pathlib.Path('dataset/test/one_letter/bold/prepared'), pathlib.Path('dataset/test/two_letters_combined/first')]
    test_ds_list = tf.data.Dataset.list_files(str(test_dirs[0]/'*/*'), shuffle=True)
    for test_dir in test_dirs[1:]:
        test_ds_list = test_ds_list.concatenate(tf.data.Dataset.list_files(str(test_dir/'*/*'), shuffle=True))

    image_count = len(test_ds_list)
    test_ds_list = test_ds_list.shuffle(image_count, reshuffle_each_iteration=True)
    test_ds = test_ds_list.take(image_count)

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


def create_model(class_count):
    model = tf.keras.Sequential([ 
        tf.keras.Input(shape=(img_size,img_size,1)),
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
        tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def train(model, train_ds, val_ds):
    MCP = ModelCheckpoint(f'{model_dir}/model.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
    ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=5,mode='max')
    RLP = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.2,min_lr=0.0001)

    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[MCP,ES,RLP])

    return history


def plot_accuracy_loss(history):
    # save graph of accuracy and loss
    epochs = history.epoch
    loss = history.history['loss']
    validation_loss = history.history['val_loss']

    plt.title('Gubitak')
    plt.xlabel('epoha')
    plt.ylabel('gubitak')
    plt.xticks(np.arange(len(epochs)), np.arange(1, len(epochs)+1))
    plt.plot(epochs, loss, c='red', label='trening')
    plt.plot(epochs, validation_loss, c='orange', label='validacija')
    plt.legend(loc='best')
    plt.savefig(f'{report_dir}/loss.pdf', format='pdf')
    plt.clf()

    acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']

    plt.title('Tačnost')
    plt.xlabel('epoha')
    plt.ylabel('tačnost')
    plt.xticks(np.arange(len(epochs)), np.arange(1, len(epochs)+1))
    plt.plot(epochs, acc, c='red', label='trening')
    plt.plot(epochs, validation_acc, c='orange', label='validacija')
    plt.legend(loc='best')
    plt.savefig(f'{report_dir}/accuracy.pdf', format='pdf')
    plt.clf()


def print_scores(model, train_ds, val_ds, test_ds):
    # evaluate the model
    scores = model.evaluate(train_ds)
    print("Train accuracy: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    scores = model.evaluate(val_ds)
    print("Validation accuracy: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    scores = model.evaluate(test_ds)
    print("Test accuracy: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def get_true_predicted_labels(model, ds):
    val_labels = []
    val_predicted = []
    for x, y in ds:
        val_predicted.extend(class_names[model.predict(x).argmax(axis=1)])
        val_labels.extend(class_names[y.numpy().argmax(axis=1)])
    return val_labels, val_predicted


def confusion_matrix(true_labels, predicted_labels, title):
    # save confusion matrix
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels, labels=class_names)
    print(confusion_matrix)
    plt.figure(figsize=(15,8))
    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Prediktovane vrednosti')
    ax.set_ylabel('Stvarne vrednosti')
    ax.set_title('Matrica konfuzije')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names, rotation='horizontal')
    plt.savefig(f'{report_dir}/{title}.pdf', format='pdf')
    plt.clf()


# functions for plotting classification report
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []

    for line in reversed(lines[2 : (len(lines) - 4)]):
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    xlabel = ''
    ylabel = ''
    xticklabels = ['Preciznost', 'Odziv', 'F1-mera']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def classification_report(true_labels, predicted_labels, title):
    # save classification report
    report = metrics.classification_report(true_labels, predicted_labels, labels=class_names)
    print(report)
    plot_classification_report(report)
    plt.savefig(f'{report_dir}/{title}.pdf', format='pdf')
    plt.clf()


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

    model = create_model(len(class_names))
    history = train(model, train_ds, val_ds)
    plot_accuracy_loss(history)

    # load model with best val accuracy
    model = tf.keras.models.load_model(f'{model_dir}/model.h5')
    print_scores(model, train_ds, val_ds, test_ds)

    val_labels, val_predicted = get_true_predicted_labels(model, val_ds)
    confusion_matrix(val_labels, val_predicted, 'confusion_matrix_val')
    classification_report(val_labels, val_predicted, 'report_val')

    test_labels, test_predicted = get_true_predicted_labels(model, test_ds)
    confusion_matrix(test_labels, test_predicted, 'confusion_matrix_test')
    classification_report(test_labels, test_predicted, 'report_test')


if __name__ == "__main__":
    main()
