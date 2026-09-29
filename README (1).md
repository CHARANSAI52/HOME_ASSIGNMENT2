# CS5720 Neural Network and Deep Learning
## Home Assignment 2

Student Name: PUPPALA CHARAN SAI  
700#: 700787150  
Course: CS5720 – Neural Network and Deep Learning  

---

## Overview

This repository contains my implementation for **Home Assignment 2** for CS5720 Neural Network and Deep Learning.

The assignment covers Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, convolution operations, edge detection, pooling operations, AlexNet, and a ResNet-like architecture.

The implementations are provided in the Jupyter Notebook:

`CS5720_Home_Assignment_2.ipynb`

---

## Assignment Objectives

The main objectives of this assignment are:

- Implement an LSTM-based RNN for text generation.
- Perform sentiment classification using an LSTM and the IMDB dataset.
- Understand convolution operations with different stride and padding configurations.
- Implement Sobel edge detection and pooling operations.
- Implement and compare simplified AlexNet and ResNet-like architectures.

---

# Question 1: Implementing an RNN for Text Generation

An LSTM-based Recurrent Neural Network is implemented to generate text one character at a time.

### Tasks Completed

- Loaded a text dataset for text generation.
- Converted the text into sequences of characters.
- Prepared the character representation for the neural network.
- Defined an LSTM-based RNN model.
- Trained the model to predict the next character.
- Generated new text by sampling characters sequentially.
- Explained the role of temperature scaling in text generation.

### Temperature Scaling

Temperature controls the randomness of the text generation process.

- A **lower temperature** makes the model more conservative and more likely to select high-probability characters.
- A **higher temperature** increases randomness and allows the model to generate more diverse text.

---

# Question 2: Sentiment Classification Using RNN

An LSTM-based sentiment classifier is implemented using the **IMDB sentiment dataset**.

### Tasks Completed

- Loaded the IMDB dataset using TensorFlow/Keras.
- Tokenized and padded the review sequences.
- Built an LSTM-based sentiment classification model.
- Trained the model to classify reviews as positive or negative.
- Generated a confusion matrix.
- Generated a classification report containing:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Discussed the importance of the precision-recall tradeoff in sentiment classification.

### Precision and Recall

Precision and recall provide different perspectives on classification performance. Precision measures how many predicted positive examples are actually positive, while recall measures how many actual positive examples are correctly identified. Considering both metrics is useful when evaluating sentiment classification.

---

# Question 3: Convolution Operations with Different Parameters

Convolution operations are implemented using a **5×5 input matrix** and a **3×3 kernel**.

The following configurations are demonstrated:

1. Stride = 1, Padding = `VALID`
2. Stride = 1, Padding = `SAME`
3. Stride = 2, Padding = `VALID`
4. Stride = 2, Padding = `SAME`

The resulting output feature maps are printed for each configuration.

### Concepts Demonstrated

- Effect of stride on the output dimensions.
- Difference between `VALID` and `SAME` padding.
- How convolution parameters affect the resulting feature map.

---

# Question 4: CNN Feature Extraction with Filters and Pooling

## Task 1: Edge Detection Using Sobel Filter

Sobel filters are used to detect edges in an image.

The implementation:

- Loads a grayscale image.
- Applies the Sobel filter in the x-direction.
- Applies the Sobel filter in the y-direction.
- Displays:
  1. Original image
  2. Sobel-X edge detection
  3. Sobel-Y edge detection

The implementation uses NumPy and OpenCV (`cv2`).

## Task 2: Max Pooling and Average Pooling

A random **4×4 matrix** is used as an input image.

The following operations are performed:

- 2×2 Max Pooling
- 2×2 Average Pooling

The original matrix and both pooled matrices are printed.

---

# Question 5: Implementing and Comparing CNN Architectures

## Task 1: Simplified AlexNet

A simplified AlexNet architecture is implemented using TensorFlow/Keras.

The model includes:

- Conv2D: 96 filters, 11×11 kernel, stride 4, ReLU
- MaxPooling: 3×3 pool size, stride 2
- Conv2D: 256 filters, 5×5 kernel, ReLU
- MaxPooling: 3×3 pool size, stride 2
- Conv2D: 384 filters, 3×3 kernel, ReLU
- Conv2D: 384 filters, 3×3 kernel, ReLU
- Conv2D: 256 filters, 3×3 kernel, ReLU
- MaxPooling: 3×3 pool size, stride 2
- Flatten layer
- Dense layer with 4096 neurons and ReLU
- Dropout with 50%
- Dense layer with 4096 neurons and ReLU
- Dropout with 50%
- Output layer with 10 neurons and Softmax

The model summary is printed after defining the architecture.

---

## Task 2: Residual Block and ResNet-like Model

A residual block is implemented with:

- Two Conv2D layers.
- 64 filters.
- 3×3 kernels.
- ReLU activation.
- A skip connection that adds the input to the output before activation.

A simple ResNet-like model is then constructed using:

- Initial Conv2D layer with 64 filters.
- 7×7 kernel.
- Stride of 2.
- Two residual blocks.
- Flatten layer.
- Dense layer with 128 neurons.
- Softmax output layer.

The model summary is printed after defining the model.

---

# Technologies and Libraries Used

The assignment uses the following technologies and Python libraries:

- Python
- Jupyter Notebook
- NumPy
- TensorFlow / Keras
- OpenCV (`cv2`)
- Scikit-learn
- Matplotlib


---

# How to Run

1. Clone or download this repository.
2. Open `CS5720_Home_Assignment_2.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
3. Install the required Python libraries if they are not already installed.
4. Run the notebook cells in order.
5. Review the generated outputs, model summaries, classification results, feature maps, and visualizations.

### Required Libraries

```bash
pip install numpy tensorflow opencv-python scikit-learn matplotlib
```

---

# Results

The notebook produces the required outputs for all five questions, including:

- Generated text from the LSTM text-generation model.
- Sentiment classification results, confusion matrix, and classification report.
- Feature maps for the four convolution configurations.
- Sobel-X and Sobel-Y edge-detection results.
- Max-pooling and average-pooling outputs.
- AlexNet model summary.
- ResNet-like model summary.

---

# Conclusion

This assignment demonstrates several fundamental deep learning concepts, including sequence modeling with LSTMs, sentiment classification, convolution operations, image feature extraction, pooling, and CNN architectures.

The implementations provide practical experience with TensorFlow/Keras and demonstrate how different neural network architectures and convolution parameters affect deep learning tasks.