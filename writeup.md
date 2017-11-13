[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Neural Network Architecture ##
My network is an FCN network consists of following bokcks. 

* Encoder
Encoder is implemented in encoder_block function.
It used SeparableConv2DKeras and BatchNormalization.
And Encoder blocks are three with depth 32,64,128.

* 1x1 convolution
After the encoders I used a 1x1 convolution layer with 256 depth.

* Decoder
After 1x1 convolution,

![alt text](./docs/misc/model.png)
