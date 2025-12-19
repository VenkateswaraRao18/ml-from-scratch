"""
Linear Regression from Scratch (No ML libraries)

Author: venkateswaraRao jannegorla
Purpose:
- Implementing Linear Regression using Gradient Descent
- Understand math + code clearly
- Beginner friendly but professional quality
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using Gradient Descent
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize hyperparameters

        learning_rate: step size for gradient descent
        epochs: number of iterations
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Train the model using Gradient Descent

        X: input features (n_samples, n_features)
        y: target values (n_samples,)
        """

        # Number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent loop
        for _ in range(self.epochs):
            # Linear prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute Mean Squared Error
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

            # Compute gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict using trained model
        """
        return np.dot(X, self.weights) + self.bias


# =======================
# Example Usage
# =======================
if __name__ == "__main__":
    # Generate simple data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)

    # Train model
    model = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Plot results
    plt.scatter(X, y, label="Actual Data")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression from Scratch")
    plt.show()

    # Print learned parameters
    print("Learned weight:", model.weights)
    print("Learned bias:", model.bias)
