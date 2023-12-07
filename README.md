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

<img width="300" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/447807a5-31ee-4294-839b-13edeb56982e">


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

With the class weighting technique , the early stop method, used to avoid overfitting, stoped the training process to the 17th epoch. In general, the plots indicate that early stopping and class weighting have likely helped prevent severe overfitting and maintained the model's generalization ability. The fluctuations in validation loss and accuracy suggest some instability, which could be due to the class weighting or particularities of the validation data, or some noice etc.

**Training and Validation Accuracy**

<img width="622" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/2a3483ea-e13d-4f35-9965-0d5c586cccf9">

- **High Accuracy:** The model is performing well at classifying the training and validation data.
- **Validation Accuracy Drop:** This reinforces the idea that the model's performance on the validation set can vary more than on the training set.
- **Convergence:** Despite the drop, the validation accuracy recovers and continues to track closely with the training accuracy, which indicates that the model is not severely overfitting by the end of training. However, the model does seem to exhibit some variability in performance on the validation set.

**Training and Validation Loss**

<img width="622" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/f8ab4e42-d762-4478-bea9-09cae8b3acdd">

- **Decreasing Loss:** Good sign that the model is learning and improving its predictions as training progresses.
- **Spikes in Validation Loss:** It seems that that certain epochs or batch updates caused the model to perform worse on the validation set., maybe because of a challenging or noisy batch data , or it could be a sign of the model beginning to overfit to the training data.
- **Stabilization:** After the initial spikes, the validation loss seems to stabilize, although it does not reach as low a value as the training loss, which may suggests overfitting.

<img width="563" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/a4b9bdba-8366-486f-b3c2-dce29560ea61">

Overall, the model demonstrates a high true positive rate, suggesting a strong performance in identifying sick patients. It also has a high true negative rate, showing it can correctly recognize healthy individuals. The relatively low number of false positives and false negatives suggests the model has good accuracy.

- **True Negatives (TN):** The model correctly identified 278 cases as healthy. These are instances where the model's prediction and the actual condition align, indicating that the model is effective at identifying healthy cases.
- **True Positives (TP):** There are 750 cases where the model correctly identified individuals as sick. This indicates a strong ability of the model to recognize the condition it's designed to detect.
- **False Positives (FP):** The model incorrectly identified 9 healthy individuals as sick. These are type I errors, where the model predicts the condition when it's not actually present.
- **False Negatives (FN):** There are 8 cases where the model failed to identify sick individuals, incorrectly classifying them as healthy.

<img width="733" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/52494a69-03c3-40c6-8c46-61931779e633">

- **True Positive Rate:** The curve starts at the top-left corner, indicating a high True Positive Rate. This means the model is correctly identifying most of the sick patients.

- **False Positive Rate:** The curve starts at the top-left corner, indicating a high True Positive Rate. This means the model is correctly identifying most of the sick patients.

<img width="257" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/0d0d6e6a-4234-448b-bb60-5824bbebff29">

The model shows excellent performance across all metrics, indicating it is highly effective in classifying individuals as either sick or healthy. Given that this is a crucial medical context, the high recall is particularly encouraging, as it implies the model is capable of correctly identifying most of the sick individuals, which is critical for a medical diagnostic tool. However, it's important to ensure these results are not due to overfitting and that the model is equally performant on unseen data.

<img width="435" alt="image" src="https://github.com/StefanatouGerasimina/COVID_Classification/assets/63111398/ceebe79a-86c5-42cc-a9c4-c8a61fb1659b">

A loss of 0.097 indicates that the model's predictions are very close to the true values. The lower the loss, the better the model is performing. This relatively low value suggests the model is doing a good job at predicting the validation set with minimal error.

#### VGG 19 with image augmentation

**Training and Validation Accuracy**

**Training and Validation Loss**
