from config import TEST_DIR, test_image
from utils.loses import FocalLoss

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

ALPHA = 0.8
GAMMA = 2

custom_objects = {'FocalLoss': FocalLoss}

trained_model = tf.keras.models.load_model('weight_metrics_and_model/trained_model.h5')



def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_dir,img)
    img = cv2.imread(rgb_path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred


rows = 1
columns = 2
for i in range(len(test_image)):
    img, pred = gen_pred(TEST_DIR, test_image[i], trained_model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
