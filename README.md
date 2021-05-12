# Face Generation
Udacity Deep Learning Nanodegree Project #4

* This repo is about how to generate new images of faces using Deep Convolutional Generative Adversarial Networks(DCGAN).
* It is implemented by using PyTorch library.
* You can refer to Original Udacity repo [here](https://github.com/udacity/deep-learning-v2-pytorch); in project-face-generation folder.

## About Deep Convolutional Generative Adversarial Networks(DCGAN)

DCGAN is one of the popular and successful network design for GAN. It mainly composes of convolution layers without max pooling or fully connected layers. It uses convolutional stride and transposed convolution for the downsampling and the upsampling. The figure below is the network design for the generator.

<img src="https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/DCGAN.png">


## Project Overview

In this project, I defined and trained a DCGAN on a CelebA dataset which contains over 200,000 celebrity faces with annotations. 

## Project Goal
The goal of this project is to get a generator network to generate _new_ images of faces using Generative Adversarial Networks (GANs) that look as realistic as possible. 
The model is trained on the CelebFaces Attributes Dataset (CelebA).
![Image of Training Dataset](https://github.com/nanditaradha/Project_Face_Generation/blob/master/assets/processed_face_data.png)

### Installation

1. For running this project on your local computer, first make sure you have git by typing `git --version` on cmd, if version number appears that means you have git installed. Go ahead and clone the repository:

```
git clone https://github.com/nanditaradha/Project_Face_Generation.git
cd Project_Face_Generation
```
2. Now please open the file with filename: dlnd_face_generation.ipynb

### Dependencies

- Make sure to create an environment for running this project using conda (you can install [Miniconda](http://conda.pydata.org/miniconda.html) for this

- Once you have Miniconda installed, please make an environment for the project like so: 
```
conda create --name face_generation python=3.6
activate face_generation
```
- Install Pytorch 
```
conda install pytorch -c pytorch
pip install torchvision
conda install matplotlib scikit-learn jupyter notebook
```
- Open the notebook to run it
```
jupyter notebook dlnd_face_generation.ipynb
```

- Install a few required pip packages, which are specified in the requirements text file.
```
pip install -r requirements.txt
```
## Project Information

### Contents

- Loading Dataset & Pre-Processed 
    - [CelebFaces Attributes Dataset (CelebA)](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)
- Create A DataLoader
- Define The Model
	- Discriminator
	- Generator
	- Initialize The Weights Of Your Network
	- Build Complete Network
- Discriminator And Generator Loss Calculation
- Optimizers Setting
- Training Performance
- Training Loss Calculation
- Generator Samples From Training 

### Results

At the end of this project, I was able to visualize the results of my trained Generator to see how it performed; my generated samples looked like fairly realistic faces with small amounts of noise.

![Image of Generated Faces](https://github.com/nanditaradha/Face-Generation/blob/master/assets/Generated_faces.png)

## Conclusion
DCGAN is one of the most popular and succesful network design for GAN. It mainly composes of convolution layers without max pooling or fully connected layers. It uses strided convolutions and transposed convolutions for the downsampling and the upsampling respectively.

