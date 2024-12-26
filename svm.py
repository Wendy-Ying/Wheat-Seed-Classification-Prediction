import matplotlib.pyplot as plt
import numpy as np

class SVM:
    # Kernel Functions
    def linear_kernel(self, x, x_support, gamma=0.5):
        return np.dot(x, x_support.T)

    def gaussian_kernel(self, x, x_support, gamma=0.5):
        squared_distance = np.sum((x[:, np.newaxis, :] - x_support) ** 2, axis=2)
        return np.exp(-gamma * squared_distance)

    kernel_funs = {'linear': linear_kernel, 'gaussian': gaussian_kernel}

    def __init__(self, kernel='gaussian', C=0.2, k=3, n_iteration=10000, lr=1e-5, early_stopping_patience=10, tol=1e-7):
        # Hyperparameters
        self.kernel_str = kernel
        self.kernel = getattr(self, kernel + '_kernel')  # Select the kernel function dynamically
        self.C = C                  # Regularization parameter
        self.k = k                  # Kernel parameter
        self.n_iteration = n_iteration  # Number of iterations for optimization
        self.lr = lr                # Learning rate
        self.early_stopping_patience = early_stopping_patience  # Patience for early stopping
        self.tol = tol              # Tolerance for early stopping

        # Training data and support vectors
        self.X, self.y = None, None
        self.alphas = None
        
        # For multi-class classification
        self.multiclass = False
        self.clfs = []

        # For storing the loss values
        self.loss_history = []

    def fit(self, X, y):
        """ Fit SVM model (Binary or Multi-Class) """
        if -1 not in y:
            y = y - 1
        
        if len(np.unique(y)) > 2:  # Multi-Class case
            self.multiclass = True
            return self.multi_fit(X, y)

        # Binary classification (0, 1 -> -1, 1)
        if set(np.unique(y)) == {0, 1}:
            y[y == 0] = -1

        # Ensure y is a Nx1 column vector (required for training)
        self.y = y.reshape(-1, 1).astype(np.double)
        self.X = X
        N = X.shape[0]

        # Compute the kernel matrix K (NxN)
        self.K = self.kernel(X, X, self.k)

        # Initialize alpha values for dual optimization (dual variables)
        alphas = np.zeros((N, 1))

        # Optimization loop (Gradient Descent for simplicity)
        best_loss = float('inf')
        no_improvement_count = 0

        for n in range(self.n_iteration):
            decision_values = self.K @ (self.y * alphas)
            gradient = np.ones((N, 1)) - decision_values
            alphas += self.lr * (self.y * gradient)
            alphas = np.clip(alphas, 0, self.C)

            # Compute the loss (Hinge loss + regularization)
            loss = np.mean(np.maximum(0, 1 - self.y * decision_values)) + 0.5 * self.C * np.sum(alphas ** 2)
            
            
            # Store the loss for plotting
            self.loss_history.append(loss)

            # Check for early stopping (if the loss has not improved for `early_stopping_patience` iterations)
            if abs(best_loss - loss) < self.tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # If no improvement for a certain number of iterations, stop early
            if no_improvement_count >= self.early_stopping_patience:
                print(f"Early stopping at iteration {n}")
                break

            best_loss = loss

        self.alphas = alphas
        self.support_vectors_mask = (self.alphas > 1e-3).flatten()  # Support vectors mask
        self.margin_support_vector_index = np.argmax((0 < self.alphas) & (self.alphas < self.C))
        
        # plot loss history
        self.plot_loss()

    def plot_loss(self):
        """ Plot the loss history """
        plt.plot(self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('SVM Training Loss')
        plt.show()

    def predict(self, X_test):
        """ Predict labels using trained model """
        
        if self.multiclass:
            return self.multi_predict(X_test)

        # Get the support vectors and their corresponding labels
        support_vectors = self.X[self.support_vectors_mask]
        support_labels = self.y[self.support_vectors_mask]
        support_alphas = self.alphas[self.support_vectors_mask]

        if support_vectors.shape[0] == 0:
            raise ValueError("No support vectors found. The model might not have converged properly.")

        # Calculate the bias term (b) using the support vectors closest to the margin
        margin_support_vector_index = np.argmax((0 < support_alphas) & (support_alphas < self.C))  # Find the first support vector in range
        if margin_support_vector_index == -1:
            margin_support_vector_index = 0  # Default to the first support vector if no valid ones are found

        margin_support_vector = support_vectors[margin_support_vector_index, np.newaxis]
        margin_label = support_labels[margin_support_vector_index]
        bias = margin_label - np.sum(support_alphas * support_labels * self.kernel(support_vectors, margin_support_vector))

        # Compute the decision function (scores for each test sample)
        decision_scores = np.sum(support_alphas * support_labels * self.kernel(support_vectors, X_test), axis=0) + bias

        # Return the predicted class (sign of the decision function)
        return np.sign(decision_scores).astype(int), decision_scores

    def get_predictions(self, X):
        """ Predict labels using trained model """
        return self.predict(X)[0] + 1

    def evaluate(self, X, y):
        """ Evaluate the model's accuracy """
        
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)

    def multi_fit(self, X, y):
        """ Multi-Class SVM using One-Versus-Rest Strategy """

        self.k = len(np.unique(y))  # Number of classes

        # Train one classifier for each class
        for i in range(self.k):
            Xs, Ys = X, y.copy()  # No need for deep copy
            Ys[Ys != i], Ys[Ys == i] = -1, +1

            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            self.clfs.append(clf)

    def multi_predict(self, X):
        """ Multi-Class Prediction by aggregating One-Versus-Rest classifiers """ 

        N = X.shape[0]
        preds = np.zeros((N, self.k))

        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)

        # Return class with highest score
        return np.argmax(preds, axis=1), np.max(preds, axis=1)
