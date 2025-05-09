# 🖼️ Image Processing and Art-Driven Augmentation

This project explores classical image processing techniques using OpenCV and neural style transfer for artistic data augmentation. This is especially helpful in computer vision projects and creative deep learning workflows. It utilizes the famous Fashion MNIST dataset with CNN as classifying model.

---

## 📚 Table of Contents

1. [Color Mapping and Image Transformations](#1-color-mapping-and-image-transformations)
2. [Reading and Displaying Images](#2-reading-and-displaying-images)
3. [Image Style Transfer with TensorFlow Hub](#3-image-style-transfer-with-tensorflow-hub)
4. [Fashion MNIST Augmentation](#4-fashion-mnist-augmentation)
5. [CNN Classification on Stylized Data](#5-cnn-classification-on-stylized-data)
6. [Evaluation with Accuracy and Confusion Matrix](#6-evaluation-with-accuracy-and-confusion-matrix)

---

## 1. Color Mapping and Image Transformations

This section sets the stage for using OpenCV (`cv2`) and `matplotlib.pyplot` to manipulate and visualize images. We start by importing required libraries and loading the test image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
```
Convert your image into various color spaces such as RGB, Grayscale, HSV, LAB, and YCrCb. These transformations are essential for preprocessing and understanding image features.

## 2. Reading and Displaying Images
```python
image = cv2.imread('00-puppy.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

## 3. Image Style Transfer with TensorFlow Hub
We apply neural style transfer using the arbitrary-image-stylization-v1 model. This transforms grayscale images into visually enhanced, artistic variants which are great for data augmentation.
```python
import tensorflow_hub as hub
style_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
```

## 4. Fashion MNIST Augmentation
We load and preprocess the Fashion MNIST dataset, convert images to RGB, and apply style transfer using an external image (e.g., "ocean.jpeg"). This introduces artistic styles into basic dataset classes for training enhancement.
```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Apply style transfer on x_train_rgb
```
## 5. CNN Classification on Stylized Data
A Convolutional Neural Network (CNN) is built using Keras to classify both the original and stylized Fashion MNIST images. This assesses the impact of augmentation through accuracy comparisons.
```python
model = Sequential([
    Conv2D(32, (4, 4), activation='relu', input_shape=(28, 28, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

## 6. Evaluation with Accuracy and Confusion Matrix
Finally, we compute the model’s accuracy on stylized and original test sets and generate a confusion matrix for detailed class-wise performance analysis.
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

