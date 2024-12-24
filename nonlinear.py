import numpy as np
import matplotlib.pyplot as plt

class NonlinearAutoencoder:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=1e-3, epochs=3000):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_dims[i + 1])))

        # List to store the loss for each epoch
        self.losses = []

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of sigmoid function
        return x * (1 - x)

    def leaky_relu(self, x, alpha=0.01):
        # Leaky ReLU activation function
        return np.maximum(alpha * x, x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        # Derivative of Leaky ReLU function
        return np.where(x > 0, 1, alpha)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            x = self.leaky_relu(np.dot(x, self.weights[i]) + self.biases[i])
            self.activations.append(x)
        # Last layer uses sigmoid activation
        x = self.sigmoid(np.dot(x, self.weights[-1]) + self.biases[-1])
        self.activations.append(x)
        return x

    def backward(self, x, y):
        # Backpropagation
        deltas = [(self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.leaky_relu_derivative(self.activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def min_max_scale(self, X):
        # Perform Min-Max normalization using NumPy
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min), X_min, X_max

    def inverse_min_max_scale(self, X_scaled, X_min, X_max):
        # Inverse Min-Max normalization
        return X_scaled * (X_max - X_min) + X_min

    def train(self, X):
        # Normalize data using Min-Max scaling
        X_scaled, self.X_min, self.X_max = self.min_max_scale(X)
        for epoch in range(self.epochs):
            self.forward(X_scaled)
            self.backward(X_scaled, X_scaled)  # Autoencoder's goal is to reconstruct input

            # Calculate the loss (Mean Squared Error)
            loss = np.mean((X_scaled - self.activations[-1])**2)
            self.losses.append(loss)  # Store the loss

    def encode(self, X):
        # Encode data
        X_scaled = (X - self.X_min) / (self.X_max - self.X_min)  # Apply Min-Max normalization
        x = X_scaled
        for i in range(len(self.weights) - 1):  # Pass through encoder part
            x = self.leaky_relu(np.dot(x, self.weights[i]) + self.biases[i])
        return x

    def decode(self, encoded_X):
        # Decode the data
        x = encoded_X
        for i in range(len(self.weights) - 1, len(self.weights)):  # Decoder part
            x = self.sigmoid(np.dot(x, self.weights[i]) + self.biases[i])
        return self.inverse_min_max_scale(x, self.X_min, self.X_max)  # Inverse normalization

    def visualize(self, X_encoded, y, n_components=2):
        # Visualize the encoded data in 2D or 3D
        fig = plt.figure()
        if n_components == 2:
            for label in np.unique(y):
                plt.scatter(
                    X_encoded[y == label, 0],
                    X_encoded[y == label, 1],
                    label=f"Class {label}",
                    alpha=0.8,
                )
            plt.xlabel("Encoded Feature 1")
            plt.ylabel("Encoded Feature 2")
            plt.title("Nonlinear Autoencoder - 2D Encoded Data")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"nonlinear_autoencoder_{n_components}d.png")
        elif n_components == 3:
            ax = fig.add_subplot(111, projection="3d")
            for label in np.unique(y):
                ax.scatter(
                    X_encoded[y == label, 0],
                    X_encoded[y == label, 1],
                    X_encoded[y == label, 2],
                    label=f"Class {label}",
                    alpha=0.8,
                )
            ax.set_xlabel("Encoded Feature 1")
            ax.set_ylabel("Encoded Feature 2")
            ax.set_zlabel("Encoded Feature 3")
            ax.set_title("Nonlinear Autoencoder - 3D Encoded Data")
            plt.legend()
            plt.savefig(f"nonlinear_autoencoder_{n_components}d.png")

    def plot_loss(self, n_components=2):
        # Plot the loss during training
        plt.figure()
        plt.plot(range(self.epochs), self.losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss during Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_loss_{n_components}d.png")
