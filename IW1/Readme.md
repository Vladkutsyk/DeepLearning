# **Butterfly Classification**
The task was to compare different architectures and use different techniques in order to beat AlexNet on the Classification task.

I decided to use three popular architectures: AlexNet, ConvNext and YOLOv11, using the architectural design of the models and the pre-trained weights.

## **AlexNet**

For the AlexNet model, I tried to train only the classification head' weights in one attempt, and some additional layers' weights in the other one. The results were quite similar, so that I decided to go further to the better architecture.

The architecture of the model is showed below.

<img width="1400" height="640" alt="image" src="https://github.com/user-attachments/assets/9774e47d-2a1c-42a0-8d56-25a3e189b853" />

## **ConvNext**

For the ConvNext, I decided to run the Base version for the first attempt. The problem, I faced, was that the number of weights restricted me in the number of layers I could train at once - as I was using Colab to train the model, so that needed to have a reasonable period per epoch. 
The solution was to:
a) implement some optimization techniques;
b) switch to Tiny ConvNext.
I tried both of them.

The architecture of the model is showed below.

<img width="895" height="243" alt="image" src="https://github.com/user-attachments/assets/77ebebf3-7a3a-4aab-9bbe-f7caa9d89084" />

## **YOLOv11 Cls**

The last architecture, which I wanted to test, was YOLO, as SOTA (state of the art). Choosing between v11 and v8, with which I was familiar, I decided to test the latter version, so that implemented cls as classification model on my data. For this run I decided to use several additional techniques:
a) built-in hyperparameters tuner;
b) self-constructed k-fold stratified crossvalidation, repeated twice.

The architecture of the model is showed below.

<img width="850" height="354" alt="image" src="https://github.com/user-attachments/assets/46064f1a-0e37-4c96-a69b-d4b23639688b" />
