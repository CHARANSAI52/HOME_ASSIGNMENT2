# CS5720 Neural Network and Deep Learning — Home Assignment 2

**Student Name:** PUPPALA CHARAN SAI  
**700#:** 700787150  
**Course:** CS5720 Neural Network and Deep Learning  
**Department:** Computer Science & Cybersecurity  
**University:** University of Central Missouri  
**Semester:** Fall 2026  

## Overview

This repository contains my implementation for CS5720 Neural Network and Deep Learning – Home Assignment 2. The assignment covers LSTM text generation, LSTM sentiment classification, convolution operations, Sobel edge detection, pooling, a simplified AlexNet, and a ResNet-like architecture.

Main notebook:

`CS5720_Home_Assignment_2.ipynb`

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (`cv2`)
- Scikit-learn
- Jupyter Notebook

---

# Question 1 — Implementing an RNN for Text Generation

The notebook uses the **Tiny Shakespeare** dataset from Andrej Karpathy's `char-rnn` repository. The first **100,000 characters** are used to keep training manageable.

### Data preparation

- Character vocabulary size: **61**
- Sequence length: **100**
- Batch size: **64**
- Character IDs are used as the sequence representation.
- Input sequences contain 100 characters and target sequences are shifted by one character.

### Model

The model contains:

- Embedding layer: 128-dimensional embeddings
- LSTM layer: 256 units
- Dense output layer with vocabulary-size output

Training configuration:

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy with logits
- Epochs: **10**

### Training result

The recorded loss decreased from **3.6584** at epoch 1 to **2.1826** at epoch 10.

### Text generation

The notebook generates 300 characters beginning with `ROMEO:` using temperatures:

- **0.2** — more predictable and repetitive
- **0.8** — balance between predictability and randomness
- **1.2** — more random and varied, but less coherent

Temperature scaling changes the distribution used when sampling the next character.

---

# Question 2 — Sentiment Classification Using RNN

The notebook uses the **IMDB dataset** from `tensorflow.keras.datasets.imdb`.

### Dataset and preprocessing

- Training samples: **25,000**
- Testing samples: **25,000**
- Vocabulary limit: **10,000 words**
- Maximum sequence length: **200**
- Padded training shape: `(25000, 200)`
- Padded testing shape: `(25000, 200)`

### Model

The model contains:

- Embedding layer: 10,000 vocabulary size, 128-dimensional embeddings
- LSTM: 64 units
- Dropout: 0.5
- Dense output: 1 neuron with sigmoid activation

Training configuration:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: **5**
- Batch size: **128**
- Validation split: **20%**

### Test results

```text
Test Loss: 0.6604546308517456
Test Accuracy: 0.5685200095176697
```

Test accuracy is approximately **56.85%**.

### Confusion matrix

```text
[[10942  1558]
 [ 9229  3271]]
```

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.5425 | 0.8754 | 0.6698 | 12,500 |
| Positive | 0.6774 | 0.2617 | 0.3775 | 12,500 |
| **Accuracy** | | | **0.5685** | **25,000** |
| Macro Avg | 0.6099 | 0.5685 | 0.5237 | 25,000 |
| Weighted Avg | 0.6099 | 0.5685 | 0.5237 | 25,000 |

### Precision-recall tradeoff

Precision measures how many predicted positive reviews are actually positive. Recall measures how many actual positive reviews are correctly identified. Changing the classification threshold changes false positives and false negatives, so the appropriate balance depends on the application.

---

# Question 3 — Convolution Operations with Different Parameters

The notebook uses a **5×5 input matrix** and a **3×3 kernel** with TensorFlow.

### Input matrix

```text
[[ 1,  2,  3,  4,  5],
 [ 6,  7,  8,  9, 10],
 [11, 12, 13, 14, 15],
 [16, 17, 18, 19, 20],
 [21, 22, 23, 24, 25]]
```

### Kernel

```text
[[ 0,  1,  0],
 [ 1, -4,  1],
 [ 0,  1,  0]]
```

### Stride = 1, Padding = VALID

```text
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
```

### Stride = 1, Padding = SAME

```text
[[  4.   3.   2.   1.  -6.]
 [ -5.   0.   0.   0. -11.]
 [-10.   0.   0.   0. -16.]
 [-15.   0.   0.   0. -21.]
 [-46. -27. -28. -29. -56.]]
```

### Stride = 2, Padding = VALID

```text
[[0. 0.]
 [0. 0.]]
```

### Stride = 2, Padding = SAME

```text
[[  4.   2.  -6.]
 [-10.   0. -16.]
 [-46. -28. -56.]]
```

These results demonstrate the effects of stride and padding on convolution output.

---

# Question 4 — CNN Feature Extraction with Filters and Pooling

## Task 1 — Sobel Edge Detection

The notebook creates a grayscale sample image using NumPy and OpenCV. The image contains a rectangle, circle, and horizontal line.

Sobel filtering is applied in:

- X direction
- Y direction

The notebook displays:

1. Original Image
2. Sobel X
3. Sobel Y

OpenCV `cv2.Sobel()` is used with a 3×3 kernel.

## Task 2 — Max Pooling and Average Pooling

A random 4×4 matrix is generated with NumPy using random seed 42.

### Original matrix

```text
[[6. 3. 7. 4.]
 [6. 9. 2. 6.]
 [7. 4. 3. 7.]
 [7. 2. 5. 4.]]
```

### Max pooled matrix

```text
[[9. 7.]
 [7. 7.]]
```

### Average pooled matrix

```text
[[6.   4.75]
 [5.   4.75]]
```

Both pooling operations use a **2×2 window with stride 2**.

---

# Question 5 — Implementing and Comparing CNN Architectures

## Task 1 — Simplified AlexNet

The notebook implements a simplified AlexNet with input shape **224×224×3**.

Architecture:

| Layer | Configuration |
|---|---|
| Conv2D | 96 filters, 11×11 kernel, stride 4, ReLU |
| MaxPooling2D | 3×3 pool, stride 2 |
| Conv2D | 256 filters, 5×5 kernel, ReLU |
| MaxPooling2D | 3×3 pool, stride 2 |
| Conv2D | 384 filters, 3×3 kernel, ReLU |
| Conv2D | 384 filters, 3×3 kernel, ReLU |
| Conv2D | 256 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 3×3 pool, stride 2 |
| Flatten | — |
| Dense | 4096 neurons, ReLU |
| Dropout | 50% |
| Dense | 4096 neurons, ReLU |
| Dropout | 50% |
| Output | 10 neurons, Softmax |

The model summary reports:

```text
Total parameters: 21,622,154
Trainable parameters: 21,622,154
Non-trainable parameters: 0
```

## Task 2 — Residual Block and ResNet-like Model

The residual block contains:

- Two Conv2D layers
- 64 filters
- 3×3 kernels
- `same` padding
- ReLU activations
- Skip connection using `layers.Add()`
- ReLU after the addition

The ResNet-like model uses:

- Input shape: **64×64×3**
- Initial Conv2D: 64 filters, 7×7 kernel, stride 2, same padding
- ReLU activation
- Two residual blocks
- Flatten
- Dense: 128 neurons, ReLU
- Output: 10 neurons, Softmax

The model summary reports:

```text
Total parameters: 8,547,210
Trainable parameters: 8,547,210
Non-trainable parameters: 0
```

---

# Repository Structure

```text
CS5720-Home-Assignment-2/
│
├── CS5720_Home_Assignment_2.ipynb
└── README.md
```

# How to Run

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

Then open:

```text
CS5720_Home_Assignment_2.ipynb
```

in Jupyter Notebook, JupyterLab, or Google Colab and run the cells from top to bottom.

The notebook downloads the Tiny Shakespeare and IMDB datasets when required.

---

# Assignment Completion

| Question | Work | Status |
|---|---|---|
| Q1 | LSTM text generation | Completed |
| Q1 | Temperature comparison | Completed |
| Q2 | IMDB preprocessing | Completed |
| Q2 | LSTM sentiment classifier | Completed |
| Q2 | Confusion matrix | Completed |
| Q2 | Classification report | Completed |
| Q3 | Four stride/padding configurations | Completed |
| Q4 | Sobel-X/Sobel-Y edge detection | Completed |
| Q4 | Max pooling | Completed |
| Q4 | Average pooling | Completed |
| Q5 | Simplified AlexNet | Completed |
| Q5 | Residual block | Completed |
| Q5 | ResNet-like model | Completed |
| Q5 | Model summaries | Completed |

# Conclusion

This assignment demonstrates practical implementations of recurrent neural networks, LSTM sequence modeling, sentiment classification, convolution operations, image edge detection, pooling, and CNN architectures.

The notebook includes the requested implementations and recorded outputs for all five questions.

---

## Student Information

**PUPPALA CHARAN SAI**  
**700#: 700787150**

**CS5720 Neural Network and Deep Learning**  
**University of Central Missouri**  
**Fall 2026**
