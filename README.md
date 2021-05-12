# Face Generation
Udacity Deep Learning Nanodegree Project #4

* This repo is about how to generate new images of faces using Deep Convolutional Generative Adversarial Networks(DCGAN).
* It is implemented by using PyTorch library.
* You can refer to Original Udacity repo [here](https://github.com/udacity/deep-learning-v2-pytorch); in project-face-generation folder.

## Project Overview

In this project, I defined and trained a DCGAN on a CelebA dataset. 

# Goal
The goal of this project is to get a generator network to generate _new_ images of faces using Generative Adversarial Networks (GANs) that look as realistic as possible. 
The model is trained on the CelebFaces Attributes Dataset (CelebA).

![Image of Celebrity Dataset](https://github.com/nanditaradha/Project_Face_Generation/blob/master/assets/processed_face_data.png)

# Installation

1. For running this project on your local computer, first make sure you have git by typing `git --version` on cmd, if version number appears that means you have git installed. Go ahead and clone the repository:

```
git clone https://github.com/nanditaradha/Project_Face_Generation.git
cd Project_Face_Generation

```
2. Now please open the file with filename: dlnd_face_generation.ipynb

# Dependencies

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

`pip install -r requirements.txt`

# Project Information

## Contents
