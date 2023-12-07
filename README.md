# COVID 19 X-RAYS CLASSIFICAITON

## Introduction

The scope of this project is to read input images of chest x-rays and predict if a patient is healthy or has covid.
The given dataset provides 1348 X-rays of healthy patients and 3875 of x-rays from patients with covid 19.

**Example of Healthy lungs**

<img width="350" height="350" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/4b512a39-11f8-4d74-a744-4a78b16d690c">


**Example of Sick lungs (Covid19)**

<img width="350" height="350" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/52b07f90-76a5-4cdd-9332-277743276571">


**Distribution of Images**

<img width="300" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/74d3b4d5-4a8c-4a59-a3da-006f229e7f86">

This suggests that within the dataset represented, the majority of the input images are classified as having a health state of 1, which means healthy. The class distribution is imbalanced, with the "1" state (Sick with Covid19) being the predominant class. This could cause a bias to the model towards the “Healthy” class. 


## Class Wheightning

In class weighting, different weights are assigned to classes in such a way that minority classes are given higher importance during the training of the model. The weights are inversely proportional to the frequency of each class, meaning classes with fewer samples are assigned higher weights. This approach ensures that the model does not become biased towards the majority class and pays adequate attention to learning from the minority class. By doing so, class weighting aims to improve the model's performance, especially its ability to correctly identify instances of the less frequent classes. So the distribution remain still the same.

## Data Augmentation


**Distribution after data augmentation**

<img width="596" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/406b28d5-da42-4f50-a042-ddde92a84978">

Augmenting data for the class of Healthy lungs with rotations, width and height shifts, horizontal flips and brightness adjustment x2 in order to reach as close as possible the percentage of images in the class of “COVID19” . The adjustments were suggested by a previous scientist research, that can be found in the following link: https://link.springer.com/article/10.1140/epjs/s11734-022-00647-x 

Original Image: 

![image](https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/90e29340-d792-4af0-8726-af2f344dbc08)

Augmented Image:

![image](https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/9564b47b-e36a-4a1b-a5d4-452e8d835fe5)

*Visual Obvious changes: Rotation, shift, filling with blank*

## XRAY Classification

### VGG 19

VGG19 is a convolutional neural network model known for its depth, consisting of 19 layers, including 16 convolutional layers, 3 fully connected layers, and 5 MaxPool layers. It's part of the VGG (Visual Geometry Group) models developed by researchers at the University of Oxford. The key characteristic of VGG19 is its uniform architecture, where it uses a stack of convolutional layers with small receptive fields (3x3 filters) followed by max pooling layers, which helps in capturing fine-grained details in images.

In summary, the depth, feature extraction capabilities, adaptability through transfer learning, proven performance in complex image recognition tasks, and relative interpretability make VGG19 a suitable choice for COVID-19 chest X-ray image classification. It can efficiently learn and distinguish the specific patterns associated with COVID-19, aiding in timely and accurate diagnosis.


**Related researches:**

- A deep learning-based COVID-19 classification from chest X-ray image: case study
- The European Physical Journal Special Topics, Appasami & Nickolas, https://link.springer.com/article/10.1140/epjs/s11734-022-00647-x
- A Deep Learning Based Covid-19 Detection Framework | IEEE Conference …, https://ieeexplore.ieee.org/document/9739806
- BND-VGG-19: A deep learning algorithm for COVID-19 identification utilizing X-ray images, https://www.sciencedirect.com/science/article/pii/S0950705122011339#b30
Etc…

#### VGG 19 with class weightning



