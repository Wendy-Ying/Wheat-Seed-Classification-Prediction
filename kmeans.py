import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, learning_rate=0.5):
        """Initialize KMeans model with learning rate."""
        self.k = k
        self.max_iters = max_iters
        self.learning_rate = learning_rate  # Learning rate (0 < learning_rate < 1)
        self.centroids = None
        self.label_mapping = {}
        self.inertia_history = []
        self.centroid_movement_history = []

    def mode(self, data):
        """Return the most frequent value in data."""
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]

    def initialize_centroids(self, X):
        """Initialize centroids using KMeans++."""
        self.centroids = np.array([X[np.random.choice(X.shape[0])]])  # Choose one random point as the first centroid

        for _ in range(1, self.k):
            # Compute distance of each point to the nearest centroid
            distances = np.array([np.min([np.linalg.norm(x - centroid) ** 2 for centroid in self.centroids]) for x in X])
            probabilities = distances / distances.sum()  # Normalize to create a probability distribution
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()

            # Select new centroid based on probability distribution
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    self.centroids = np.vstack((self.centroids, X[i]))
                    break

    def fit(self, X, y):
        """Fit the model to the data."""
        self.initialize_centroids(X)
        for n in range(self.max_iters):
            # Assign each point to the nearest centroid
            closest_centroids = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

            # Update centroids (with learning rate mechanism)
            new_centroids = np.array([X[closest_centroids == i].mean(axis=0) if X[closest_centroids == i].size else X[np.random.choice(X.shape[0])] for i in range(self.k)])

            # Apply learning rate to smooth centroid update
            self.centroids = self.centroids * (1 - self.learning_rate) + new_centroids * self.learning_rate

            # Compute inertia (sum of squared distances to centroids)
            inertia = np.sum([np.sum(np.linalg.norm(X[closest_centroids == i] - self.centroids[i], axis=1) ** 2) for i in range(self.k)])
            self.inertia_history.append(inertia)

            # Compute centroid movement (Euclidean distance between old and new centroids)
            centroid_movement = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroid_movement_history.append(centroid_movement)

            # If centroid movement is very small, stop the algorithm
            if centroid_movement < 1e-6:
                print(f"Converged at iteration {n}")
                break

        # Map the predicted labels to true labels using the training set
        train_predictions = self.predict(X)
        self.label_mapping = self.map_labels(y, train_predictions)

    def predict(self, X):
        """Predict the closest cluster for each data point."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def map_labels(self, true_labels, predicted_labels):
        """Map cluster labels to true labels using voting (mode) of true labels."""
        label_mapping = {}

        for cluster in range(self.k):
            # Get the true labels of the samples assigned to the current cluster
            cluster_labels = true_labels[predicted_labels == cluster]

            if cluster_labels.size > 0:
                most_common_label = self.mode(cluster_labels)  # Get the most frequent label
                label_mapping[cluster] = most_common_label
            else:
                label_mapping[cluster] = -1  # Placeholder for empty clusters
        
        return label_mapping

    def map_predictions(self, predicted_labels):
        """Map predicted cluster labels to true labels."""
        return np.array([self.label_mapping[cluster] for cluster in predicted_labels])

    def predict_and_map(self, X):
        """Predict and map to true labels."""
        predicted_labels = self.predict(X)
        return self.map_predictions(predicted_labels)
    
    def plot_metrics(self):
        """Plot the Inertia and Centroid Movement metrics."""
        # Plot Inertia over iterations
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.inertia_history, label="Inertia")
        plt.xlabel('Iterations')
        plt.ylabel('Inertia')
        plt.title('Inertia (Cluster Loss) Over Iterations')
        plt.legend()

        # Plot Centroid Movement over iterations
        plt.subplot(1, 2, 2)
        plt.plot(self.centroid_movement_history, label="Centroid Movement", color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Centroid Movement')
        plt.title('Centroid Movement Over Iterations')
        plt.legend()

        plt.tight_layout()
        plt.savefig("Metrics")
        plt.show()
