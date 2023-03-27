from keras import layers
from keras import backend as K
from keras.datasets import fashion_mnist
from PIL import Image

import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = keras.models.load_model("./data/variant1_model.h5")

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

def outputs(model, images, savepath):
    extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

    features = extractor(images[:1])
    
    fig = plt.figure(figsize=(50, 20))
    n = len(features)
    p = 1
    for i in range(n):
        if len(features[i].shape) == 4:
            p += 1
    for i, feature in enumerate(features):
        if len(feature.shape) == 4:
            plt.subplot(1, p, i+1)
            plt.imshow(feature[0, :, :, 0], cmap='viridis')
        if i == n-1:
            plt.subplot(1, p, p)
            plt.imshow(feature, cmap='viridis')
        plt.axis('off')
        plt.title(model.layers[i].name, fontsize=36)
    plt.savefig(savepath)

outputs(model, test_images, 'variant1.pdf')