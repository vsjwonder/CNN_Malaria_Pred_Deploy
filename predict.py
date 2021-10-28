#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:47:05 2020

@author: Vivek
"""
import tensorflow as tf
#from keras.models import load_model
import tensorflow as tf
#import os
from time import time
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
        #model = load_model('model_own.h5')
        tflite_model_path = "model_own.tflite"

        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on input data.
        input_shape = input_details[0]['shape']


        # summarize model
        # model.summary()
        imagename = self.filename
        test_image = load(imagename)

        input_data = test_image

        interpreter.set_tensor(input_details[0]['index'], input_data)

        time_before = time()
        interpreter.invoke()
        time_after = time()
        total_tflite_time = time_after - time_before
        #print("Total prediction time for tflite without opt model is: ", total_tflite_time)

        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
        #print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")

        # result = model.predict(test_image)
        result = interpreter.get_tensor(output_details[0]['index'])
        result = np.argmax(result, axis=1)

        #result = model.predict(test_image)
        #result = np.argmax(result, axis=1)

        if result > 0.5:
            prediction = 'The cell is: Non infected'
            return [prediction]
        else:
            prediction = 'The cell is: Infected'
            return [prediction]
