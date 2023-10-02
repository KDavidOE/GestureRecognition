# GestureRecognition
This project provides a software to control PowerPoint presentations using hand gestures.

The main goal is to create a robust system that can handle difficulties such as dynamic backgrounds.
Dynamic backgrounds could involve the presence of multiple moving hands in a frame, different skin colors, insufficient lighting, and shadows.

The solution includes the following components:

- A Convolutional Neural Network
    - Based on 4 block layers
    - Three blocks constist of max-pooling and convolutional layers
    - Final block is made out of a flatten and a dense layer
- A Conditional Generative Adverserial Network.
    - The generator is a U-net decoder
    - The discriminator is a "Markovian patch-discriminator"
- Various image processing solutions, such as
    - Optical flow
    - Google's MediaPipe
    - Gauss filter, color thresholding etc.
- The GUI is implemented using PySimpleGui, and the system calls performed with PyWin32
- The input data dimensions: 64 x 64 x 3

Class diagram:

![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/bccdeef2-0217-4e10-8d6d-41b0c61e01ca)

Segmentation result:

| ![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/b1c8ff6a-6ab4-4475-a2a3-4d9e7259b6ea)|
|:--:|
| Combined result of the Optical flow, MediaPipe and skin-color segmentation methods |

# Training CNN and C-GAN on custom ASL dataset.

Creating the input dataset for a GAN network using image-pairs:

![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/da7241c3-0623-4a2f-ab6c-aade1a4c1c91)|
|:--:|
| Applied masking methods on dataset to train C-GAN |

Testing result on real data:

| ![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/7fecfa4b-7828-49b3-bf67-2115c9f49749)| 
|:--:| 
| Input images |
|![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/31125cf9-d4e5-4423-8dee-2bad8322b58e)|
| Images regenerated by C-GAN |

Evaluating the results based on each gesture:

| ![image](https://github.com/KDavidOE/GestureRecognition/assets/101677036/79281d11-e454-43ad-b3e0-75835cdc5b70)|
|:--:|
| Final NN metrics |


# References
- B. Jason, „Implementation of pix2pix with keras.” https:// machinelearningmastery.com/how-to-implement-pix2pix- gan-models-from-scratch-with-keras/
- D. Dahmani, M. Cheref, and S. Larabi, „Zero-sum game theory model for segmenting skin regions,” Image and Vision Computing, vol. 99, p. 103925, 2020
- O. Ronneberger, P. Fischer, and T. Brox, „U-net: Convolutional networks for bio- medical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241, Springer, 2015
- Google, „On-device, real-time hand tracking with mediapipe.” https: //ai.googleblog.com/2019/08/on-device-real-time-hand- tracking-with.html
# Note
- If the network models or the datasets are needed, contact me.
