import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
imgSize = 28

def processImage(img):
    img = cv2.resize(img, (imgSize, imgSize))
    return img

model1 = tf.keras.models.load_model('./model/two_letters/my_model')

def test_merged():
    for letter in letters:
        n = 0
        good = 0
        
        for filename in os.listdir('./dataset/validation/two_letters_combined/first/' + letter):
            if filename.endswith(".png") or filename.endswith(".jpg"): 
                img = cv2.imread('./dataset/validation/two_letters_combined/first/' + letter + '/' + filename, 0)

                n += 1
                X = []
                X.append(img)
                X = np.array(X).reshape(len(X), imgSize,imgSize,1)

                yhat = model1.predict(X)[0]
                arg = np.argmax(yhat)
                if (letters[arg] == letter):
                    good += 1

        print(letter, good/n)

test_merged()
