## Introduction 
This is the repo of the paper [Marker-less Kendo Motion Prediction Using High-speed Dual-camera System and LSTM Method](https://ieeexplore.ieee.org/document/9863303).
To build this project, you have to correctly install the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) at first since the detecting part is based on it.

## Functions
By this project, the pre-trained model of the Kendo motions classification model will be loaded and then human motions will be monitoring then compute the possible attack pattern.
In addition, we applied the UR robotic arm to execute the defenses according to the predicted attack patterns.

## Speed
The running speed is related to the hardwares. 
In my case, with 2 M5000 GPU accelerated, the frequency can be up to 120 fps.
A simple python version program running on CPU will be released later.

## Others
In this project, we used the ximea cameras to capture the human motions. You have to modify the related codes to make this project suit your camera models.

