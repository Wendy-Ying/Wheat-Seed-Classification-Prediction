import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.projection_matrix = None

    def fit(self, X):
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        # Select the top n_components eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.projection_matrix = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(self.n_components)])

    def transform(self, X):
        # Project data onto the principal components
        return X.dot(self.projection_matrix)

    def visualize(self, X, y):
        X_pca = self.transform(X)
        plt.figure()
        colors = ['r', 'b', 'g']
        markers = ['s', 'x', 'o']
        for label, color, marker in zip(np.unique(y), colors, markers):
            plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], c=color, label=label, marker=marker)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('PCA - Projected Data')
        plt.legend()
        plt.savefig('pca_2d.png')
        plt.show()

    def visualize_3d(self, X, y):
        X_pca = self.transform(X)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'b', 'g']
        markers = ['s', 'x', 'o']
        
        for label, color, marker in zip(np.unique(y), colors, markers):
            ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], X_pca[y == label, 2],
                       c=color, label=f'Class {label}', marker=marker)
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title('PCA - Projected Data (3D)')
        ax.legend()
        plt.savefig('pca_3d.png')
        plt.show()

    def plot_explained_variance(self):
        total_variance = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.figure()
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA - Explained Variance')
        plt.savefig('pca_explained_variance.png')
        plt.show()
