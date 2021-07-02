# Code is taken from: https://github.com/eriklindernoren/Keras-GAN

from __future__ import print_function, division

import tensorflow as tf
import os, cv2,tqdm, random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Activation, Embedding, ZeroPadding2D, Lambda
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys
import pathlib

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DCGAN():
    def __init__(self, letter):
        self.letter = letter

        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = []

        base_dir = f'../dataset/train/one_letter/medium/{self.letter}'
        for img_name in tqdm.tqdm(os.listdir(base_dir)):
            img_path = os.path.join(base_dir, img_name)
            img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            X_train.append(np.array(img))

        base_dir = f'../dataset/train/one_letter/normal/{self.letter}'
        for img_name in tqdm.tqdm(os.listdir(base_dir)):
            img_path = os.path.join(base_dir, img_name)
            img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            X_train.append(np.array(img))

        base_dir = f'../dataset/train/one_letter/bold/{self.letter}'
        for img_name in tqdm.tqdm(os.listdir(base_dir)):
            img_path = os.path.join(base_dir, img_name)
            img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            X_train.append(np.array(img))

        X_train = np.array(X_train).reshape(len(X_train), 28,28,1)

        index = list(range(len(X_train)))
        random.shuffle(index)

        X_train = X_train[index]

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"training_images/{self.letter}/{epoch}")
        plt.close()

    def save_imgs_good(self):
        n = 0
        while n < 1000:
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            valid = self.discriminator(gen_imgs)[0]
            if valid >= 0.5:
                n += 1
                name = f'{n}-{valid}.png'
                # Rescale images 0 - 1
                gen_imgs = 0.5 * gen_imgs + 0.5
                img = gen_imgs[0]
                cv2.imwrite(f'out/{self.letter}/{name}', img*255)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 dcgan.py [letter] [train,generate]")

    letter = sys.argv[1]
    train = sys.argv[2] == 'train'

    dcgan = DCGAN(letter)

    if train:
        pathlib.Path(f'model/{letter}').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'training_images/{letter}').mkdir(parents=True, exist_ok=True)
        dcgan.train(epochs=4000, batch_size=32, save_interval=50)
        dcgan.generator.save(f"model/{letter}/generator.h5")
        dcgan.discriminator.save(f"model/{letter}/discriminator.h5")
    else:
        pathlib.Path(f'out/{letter}').mkdir(parents=True, exist_ok=True)
        dcgan.generator = tf.keras.models.load_model(f'model/{letter}/generator.h5')
        dcgan.discriminator = tf.keras.models.load_model(f'model/{letter}/discriminator.h5')
        dcgan.save_imgs_good()

