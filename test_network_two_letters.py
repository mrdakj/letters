import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
imgSize = 28

def processImage(img):
    img = cv2.resize(img, (imgSize, imgSize))
    img = img / 255.0
    return img

model1 = tf.keras.models.load_model('./model/two_letters/first/model.h5')
# os.mkdir('wrong')

def test_letter(letter):
    # os.mkdir('wrong/' + letter)
    def recognize(img,n,good, filename):
        n += 1
        X = []
        img = processImage(img)
        X.append(img)
        X = np.array(X).reshape(len(X), imgSize,imgSize,1)

        yhat = model1.predict(X)[0]
        arg = np.argmax(yhat)
        if (letters[arg] == letter):
            good += 1
        # else:
            # cv2.imwrite('wrong/' + letter + '/' + filename, img)
            # print(letters[arg])
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
        return n,good

    n = 0
    good = 0
    for filename in os.listdir('./dataset/validation/two_letters/' + letter):
        if filename.endswith(".png") or filename.endswith(".jpg"): 
            img = cv2.imread('./dataset/validation/two_letters/' + letter + '/' + filename, 0)
            n, good = recognize(img, n, good, filename)

    print(letter, good/n)

test_letter('a')
test_letter('b')
test_letter('c')
test_letter('d')
test_letter('e')
test_letter('f')
test_letter('g')
test_letter('h')
test_letter('k')
test_letter('l')
test_letter('m')
test_letter('n')

def test_merged():
    for letter in letters:
        n = 0
        good = 0
        
        for filename in os.listdir('./dataset/validation/two_letters_combined/first/' + letter):
            if filename.endswith(".png") or filename.endswith(".jpg"): 
                img = cv2.imread('./dataset/validation/two_letters_combined/first/' + letter + '/' + filename, 0)

                n += 1
                X = []
                img = img / 255.0
                X.append(img)
                X = np.array(X).reshape(len(X), imgSize,imgSize,1)

                yhat = model1.predict(X)[0]
                arg = np.argmax(yhat)
                if (letters[arg] == letter):
                    good += 1
                # else:
                #     print(letters[arg])
                #     cv2.imshow('img',img)
                #     cv2.waitKey(0)


        print(letter, good/n)

test_merged()
