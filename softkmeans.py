import numpy as np
import matplotlib.pyplot as plt

class SoftKMeans:
    def __init__(self, k=3, max_iters=100, m=2, epsilon=1e-6):
        """
        Initialize Soft KMeans
        :param k: Number of clusters
        :param max_iters: Maximum number of iterations
        :param m: Membership weighting coefficient (typically 2)
        :param epsilon: Small constant to avoid division by zero
        """
        self.k = k
        self.max_iters = max_iters
        self.m = m  # Membership weighting coefficient
        self.epsilon = epsilon  # Small constant to avoid division by zero
        self.centroids = None
        self.membership = None
        self.inertia_history = []
        self.centroid_movement_history = []

    def mode(self, data):
        """Return the most frequent value in the data"""
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]

    def initialize_membership(self, X):
        """Randomly initialize membership and normalize"""
        membership = np.random.rand(X.shape[0], self.k)
        return membership / np.sum(membership, axis=1, keepdims=True)

    def fit(self, X, y):
        """Train the model"""
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]  # Random initialization of centroids
        self.membership = self.initialize_membership(X)

        for n in range(self.max_iters):
            # Update membership
            self.membership = self.update_membership(X)

            # Update centroids
            new_centroids = self.update_centroids(X)

            # Calculate inertia
            inertia = self.calculate_inertia(X, new_centroids)
            self.inertia_history.append(inertia)

            # Calculate centroid movement
            centroid_movement = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroid_movement_history.append(centroid_movement)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {n}")
                break

            self.centroids = new_centroids

        # After training, map the labels
        train_predictions = self.predict(X)
        self.label_mapping = self.map_labels(y, train_predictions)

    def update_membership(self, X):
        """Update the membership values for each point"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        distances = np.maximum(distances, self.epsilon)  # Avoid division by zero
        membership = 1 / (distances ** 2)
        return membership / np.sum(membership, axis=1, keepdims=True)  # Normalize the membership

    def update_centroids(self, X):
        """Update centroids based on membership and a learning rate"""
        alpha = 0.5  # Learning rate
        new_centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            weight = self.membership[:, k] ** self.m
            updated_centroid = np.sum(weight[:, np.newaxis] * X, axis=0) / np.sum(weight)
            # Apply learning rate
            new_centroids[k] = alpha * updated_centroid + (1 - alpha) * self.centroids[k]
        return new_centroids

    def calculate_inertia(self, X, centroids):
        """Calculate the inertia (sum of squared distances to the nearest centroid)"""
        inertia = np.sum([np.sum(np.linalg.norm(X[self.membership[:, k] > 0.5] - centroids[k], axis=1) ** 2)
                          for k in range(self.k)])
        return inertia

    def predict(self, X):
        """Predict the cluster assignment for each data point"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def map_labels(self, true_labels, predicted_labels):
        """Map cluster labels to true labels based on the majority vote"""
        label_mapping = {}
        for cluster in range(self.k):
            cluster_labels = true_labels[predicted_labels == cluster]
            most_common_label = self.mode(cluster_labels) if cluster_labels.size > 0 else -1
            label_mapping[cluster] = most_common_label
        return label_mapping

    def map_predictions(self, predicted_labels):
        """Map predicted cluster labels to true labels"""
        return np.array([self.label_mapping[cluster] for cluster in predicted_labels])

    def predict_and_map(self, X):
        """Predict and map the cluster labels to true labels"""
        predicted_labels = self.predict(X)
        return self.map_predictions(predicted_labels)

    def plot_metrics(self):
        """Visualize inertia and centroid movement over iterations"""
        plt.figure(figsize=(10, 5))

        # Plot inertia
        plt.subplot(1, 2, 1)
        plt.plot(self.inertia_history, label="Inertia")
        plt.xlabel('Iterations')
        plt.ylabel('Inertia')
        plt.title('Inertia (Cluster Loss) Over Iterations')
        plt.legend()

        # Plot centroid movement
        plt.subplot(1, 2, 2)
        plt.plot(self.centroid_movement_history, label="Centroid Movement", color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Centroid Movement')
        plt.title('Centroid Movement Over Iterations')
        plt.legend()

        plt.tight_layout()
        plt.savefig("Metrics")
        plt.show()


