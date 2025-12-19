# Linear Regression From Scratch (Complete Guide)

Author: Lakshmi Narayana Nallamothu  
GitHub Ready • Beginner → Professional Level

---

## 1. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to model the relationship between:

- **Independent variable(s)** X
- **Dependent variable** y

The goal is to find a straight line that best fits the data.

---

## 2. Mathematical Model

### Simple Linear Regression

\[
y = mx + b
\]

Where:

- m = slope (weight)
- b = bias (intercept)

### Multiple Linear Regression

\[
y = w_1x_1 + w_2x_2 + ... + w_nx_n + b
\]

Vector form:
\[
\hat{y} = Xw + b
\]

---

## 3. Why Do We Need a Loss Function?

To measure how wrong our predictions are.

### Mean Squared Error (MSE)

\[
MSE = L=n1​i=1∑n​(yi​−y^​i​)2
\]

- Penalizes large errors
- Differentiable → good for optimization

---

## 4. Optimization: Gradient Descent

We minimize MSE by updating weights step by step.

### Gradients

\[
\frac{\partial MSE}{\partial w} = -\frac{2}{n} X^T (y - \hat{y})
\]

\[
\frac{\partial MSE}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})
\]

### Update Rules

\[
w = w - \alpha \cdot dw
\]

\[
b = b - \alpha \cdot db
\]

Where:

- α = learning rate

---

## 5. Algorithm Steps (Plain English)

1. Initialize weights = 0, bias = 0
2. Predict y using current parameters
3. Calculate error (loss)
4. Compute gradients
5. Update parameters
6. Repeat until convergence

---

## 6. Python Implementation (From Scratch)

```python
y_pred = np.dot(X, weights) + bias
loss = np.mean((y - y_pred) ** 2)
```
