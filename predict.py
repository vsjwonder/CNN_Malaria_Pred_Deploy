#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:47:05 2020

@author: Vivek
"""
from keras.models import load_model

from PIL import Image
import numpy as np
from skimage import transform


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


class malaria:
    def __init__(self, filename):
        self.filename = filename

    def predictionmalaria(self):
        # load model
        model = load_model('model_own.h5')

        # summarize model
        # model.summary()
        imagename = self.filename
        test_image = load(imagename)
        result = model.predict(test_image)
        result = np.argmax(result, axis=1)

        if result > 0.5:
            prediction = 'The cell is: Non infected'
            return [prediction]
        else:
            prediction = 'The cell is: Infected'
            return [prediction]
