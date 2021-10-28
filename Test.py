import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import os
from time import time

#model = load_model('model_own.h5')
tflite_size = os.path.getsize("model_own.tflite")/1048576  #Convert to MB
print("tflite Model without opt. size is: ", tflite_size, "MB")
#Not optimized (file size = 540MB). Taking about 0.5 seconds for inference
tflite_model_path = "model_own.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Load image
from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
test_image = load(r'F:\AI\iNeuron\Online_deployment\VSJ_Malaria_Detection\Sample_Image\Infected_Cell.png')

input_data = test_image

interpreter.set_tensor(input_details[0]['index'], input_data)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_time = time_after - time_before
print("Total prediction time for tflite without opt model is: ", total_tflite_time)

output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")


#result = model.predict(test_image)
result = interpreter.get_tensor(output_details[0]['index'])
result = np.argmax(result, axis=1)
print(result)

#from keras.preprocessing.image import ImageDataGenerator
'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'F:\AI\DeepLearn_iNeuron\ImageAnalysis1st\DCData\Parent\Training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'F:\AI\DeepLearn_iNeuron\ImageAnalysis1st\DCData\Parent\Testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
training_set.class_indices '''
if result > 0.5:
    prediction = 'Uninfected'
    print(prediction)
    #return [{"image": prediction}]
else:
    prediction = 'Infected'
    print(prediction)
    #return [{"image": prediction}]