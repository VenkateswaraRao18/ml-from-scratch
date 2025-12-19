# 📈 Linear Regression From Scratch (NumPy)

A **from-scratch implementation of Linear Regression using NumPy**, built to deeply understand the mathematics behind gradient descent while maintaining **numerical stability** and **professional code quality**.

This project avoids high-level ML libraries (such as `scikit-learn`) and focuses on **core concepts, clean implementation, and best practices**.

---

## 🚀 Overview

Linear Regression models the relationship between input features and a target variable using a linear function:

$$
y = wx + b
$$

Where:

- **w** → weight (slope)
- **b** → bias (intercept)

The model is trained by minimizing **Mean Squared Error (MSE)** using **Gradient Descent**.

---

## 🎯 Objectives

- Implement Linear Regression **from scratch**
- Understand **loss functions and gradient computation**
- Avoid **numerical overflow and instability**
- Use **vectorized NumPy operations**
- Build code suitable for **professional review and interviews**

---

## 🧠 Mathematical Foundation

### Model Prediction

$$
\hat{y} = wx + b
$$

---

### Loss Function (Mean Squared Error)

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

- \( y_i \) is the actual value
- \( \hat{y}\_i \) is the predicted value
- \( n \) is the number of samples

---

## 📐 Gradient Computation

To minimize the loss, we compute partial derivatives of the loss function.

### Gradient with respect to weight \( w \)

$$
\frac{\partial L}{\partial w} =
\frac{2}{n} \sum_{i=1}^{n} x_i (\hat{y}_i - y_i)
$$

---

### Gradient with respect to bias \( b \)

$$
\frac{\partial L}{\partial b} =
\frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

---

## 🔄 Parameter Update Rules (Gradient Descent)

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

Where:

- \( \alpha \) is the **learning rate**

---

## 🧪 Dataset Generation

```python
import numpy as np

np.random.seed(0)

X = np.random.rand(100, 1) * 10
y = 3 * X + 5 + np.random.randn(100, 1)
```

## ⚙️ Feature Scaling (Critical for Stability)

Feature normalization prevents numerical overflow and speeds up convergence:

```python

X = (X - X.mean()) / X.std()

```

## 🏗️ Model Initialization

```python
w = 0.0
b = 0.0
learning_rate = 0.01
n = len(X)

```

## 🔁 Training Loop (Gradient Descent)

```python
for epoch in range(1000):

    # Forward pass
    y_pred = w * X + b

    # Error
    error = y_pred - y

    # Gradients
    dw = (2 / n) * np.sum(X * error)
    db = (2 / n) * np.sum(error)

    # Parameter update
    w -= learning_rate * dw
    b -= learning_rate * db

    # Monitor loss
    if epoch % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

```

## 📉 Loss Interpretation

- High loss → poor predictions

- Decreasing loss → model learning correctly

- Exploding loss / NaNs → learning rate too high or missing normalization

## 🧮 Vectorized NumPy Implementation

A more scalable and NumPy-idiomatic version:

```python
w = np.zeros((1,))
b = 0.0

for _ in range(1000):
    y_pred = X.dot(w) + b
    error = y_pred - y

    dw = (2 / n) * X.T.dot(error)
    db = (2 / n) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

```

## 📊 Validation Check

```python
initial_loss = np.mean((y - (0 * X + 0)) ** 2)
final_loss = np.mean((y - (w * X + b)) ** 2)

print("Initial Loss:", initial_loss)
print("Final Loss:", final_loss)

```
