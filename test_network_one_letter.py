import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
imgSize = 28

def processImage(img):
    img = img / 255.0
    return img

model1 = tf.keras.models.load_model('model/one_letter/Best_points.h5')
# os.mkdir('wrong')

all_images = 0
all_good = 0

def test_letter(letter):
    global all_images
    global all_good
    # os.mkdir('wrong/' + letter)
    def recognize(img, n, good, filename):
        n += 1
        X = []
        X.append(processImage(img))
        X = np.array(X).reshape(len(X), imgSize,imgSize,1)

        yhat = model1.predict(X)[0]
        arg = np.argmax(yhat)
        if (letters[arg] == letter):
            good += 1
        # else:
        #     cv2.imwrite('wrong/' + letter + '/' + filename, img)
        #     print(arg, letters[arg])
        #     cv2.imshow('img',img)
        #     cv2.waitKey(0)
        return n,good

    n = 0
    good = 0
    for filename in os.listdir('./dataset/validation/one/' + letter):
        if filename.endswith(".png") or filename.endswith(".jpg"): 
            img = cv2.imread('./dataset/validation/one/' + letter + '/' + filename, 0)
            n, good = recognize(img, n, good, filename)
            all_images += n
            all_good += good

    print(letter, good/n)

for letter in letters:
    test_letter(letter)

print(all_good/all_images)
