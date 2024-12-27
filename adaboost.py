import numpy as np

# Define a simple decision stump classifier
class DecisionStump:
    def fit(self, X, y, sample_weights):
        self.best_feature = None
        self.best_threshold = None
        self.best_polarity = None
        self.best_error = float('inf')
        m, n = X.shape
        
        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(m)
                    predictions[polarity * X[:, feature] < polarity * threshold] = -1
                    
                    errors = (predictions != y)
                    weighted_error = np.sum(sample_weights * errors)
                    
                    if weighted_error < self.best_error:
                        self.best_error = weighted_error
                        self.best_feature = feature
                        self.best_threshold = threshold
                        self.best_polarity = polarity

    def predict(self, X):
        m = X.shape[0]
        predictions = np.ones(m)
        feature_values = X[:, self.best_feature]
        threshold = self.best_threshold
        polarity = self.best_polarity
        
        predictions[polarity * feature_values < polarity * threshold] = -1
        return predictions

# Define AdaBoost class for multiclass classification
class AdaBoostMulticlass:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
    
    def fit(self, X, y, num_classes):
        m, n = X.shape
        
        # Use One-vs-All strategy, train a classifier for each class
        for class_id in range(1, num_classes + 1):  # Classes start from 1
            print(f"Training classifier for class {class_id}")
            # Convert labels to the current class vs other classes
            y_binary = np.where(y == class_id, 1, -1)
            sample_weights = np.ones(m) / m  # Initialize sample weights
            
            classifiers = []
            alphas = []
            
            # Train multiple weak classifiers
            for _ in range(self.n_estimators):
                stump = DecisionStump()
                stump.fit(X, y_binary, sample_weights)
                predictions = stump.predict(X)
                errors = (predictions != y_binary)
                weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
                
                alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
                
                # Update sample weights
                sample_weights *= np.exp(-alpha * y_binary * predictions)
                sample_weights /= np.sum(sample_weights)
                
                classifiers.append(stump)
                alphas.append(alpha)
            
            # Save the classifiers and corresponding alphas
            self.models.append((classifiers, alphas))
    
    def predict(self, X):
        m = X.shape[0]
        num_classes = len(self.models)
        
        # Initialize the weighted vote for each class
        predictions = np.zeros((m, num_classes))
        
        # Perform voting for each class
        for class_id in range(num_classes):
            classifiers, alphas = self.models[class_id]
            for stump, alpha in zip(classifiers, alphas):
                # Predict results from each weak classifier
                stump_predictions = stump.predict(X)
                predictions[:, class_id] += alpha * stump_predictions
        
        # Choose the class with the most votes as the final prediction
        final_predictions = np.argmax(predictions, axis=1) + 1  # Output class starts from 1
        
        return final_predictions
