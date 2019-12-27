from __future__ import print_function

import cv2
import numpy as np
import keras
import sys
import os
from drawille import Canvas, getTerminalSize

def horizontal_kernel_initializer(shape, dtype=None):
    print(shape)
    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype).reshape(shape)
    print(kernel.shape)
    
    return kernel
    
def vertical_kernel_initializer(shape, dtype=None):
    kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype).reshape(shape)
    
    return kernel

def preprocess_img(path_to_img, height=150):
    if not os.path.exists(img_path):
        print("Image path does to exist.")
        exit
    img = cv2.imread(path_to_img, 0)

    width = int(img.shape[1]/float(img.shape[0]) * height)

    img = cv2.resize(img,(width,height))

    return img

def get_model(height, width):
    inputs = keras.layers.Input(shape=(height,width,1,))

    hor = keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', kernel_initializer=horizontal_kernel_initializer)(inputs)
    ver = keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', kernel_initializer=vertical_kernel_initializer)(inputs)

    outputs = keras.layers.add([hor, ver])

    model = keras.models.Model(inputs= inputs, outputs = outputs)

    return model

def get_binary_img(img):
    height, width = img.shape
    img = img.reshape(1, height, width, 1)
    result = np.clip(get_model(height,width).predict(img),0,255).astype('uint8').reshape(height,width)
    (thresh, result) = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return result

def get_output(img):
    """
    Input: Image as numpy array.
    """
    height, width = img.shape
    c = Canvas()
    with open("out.txt",'w') as f:
        for h in range(height):
            for w in range(width):
                if img[h,w] == 255:
                    c.set(w,h)
        f.write(c.frame())
        f.write("\n")
        return c.frame()

if __name__ == '__main__':

    img_path = input("Path to image: \n")

    preprocessed_img = preprocess_img(img_path, 150)
    binary_img = get_binary_img(preprocessed_img)
    output = get_output(binary_img)
    os.system("echo " + output)
