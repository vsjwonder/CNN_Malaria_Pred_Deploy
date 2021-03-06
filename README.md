# CNN_Malaria_Pred_Deploy
This repository was created to leverage potential of AI for detection of Malaria infection. Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected mosquitoes.

If an infected mosquito bites you, parasites carried by the mosquito enter your blood and start destroying oxygen-carrying red blood cells (RBC). Typically, the first symptoms of malaria are similar to a virus like the flu and they usually begin within a few days or weeks after the mosquito bite. However, these deadly parasites can live in your body for over a year without causing symptoms, and a delay in treatment can lead to complications and even death. Therefore, early detection can save lives.

A bottleneck in malaria diagnosis
Microscopic examination of blood is the best known method for diagnosis of malaria (Gold standard for detection). A patient’s blood is smeared on a glass slide and stained with a contrasting agent that facilitates identification of parasites within red blood cells. A trained clinician examines 20 microscopic fields of view at 100 X magnification, counting red blood cells that contain the parasite out of 5,000 cells (WHO protocol).

https://miro.medium.com/max/875/1*VWxRC2BePykk3xVVEDzbdg.png

As you can imagine, manually counting 5,000 cells is a slow process. This can easily burden clinic staff, especially where outbreaks occur. Therefore, I wanted to determine how image analysis and machine learning could reduce the burden on clinicians and help prioritize patients.

Deep learning models, or more specifically convolutional neural networks (CNNs), have proven very effective in a wide variety of image analysis and prediction tasks.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 50176)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 500)               25088500  
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 1002      
=================================================================
Total params: 25,100,046
Trainable params: 25,100,046
Non-trainable params: 0
_________________________________________________________________
