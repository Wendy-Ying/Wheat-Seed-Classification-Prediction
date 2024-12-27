from load import *
from kmeans import *
from softkmeans import *
from pca import *
from nonlinear import *
from multi_layer_perceptron import *
from svm import *
from adaboost import *
from evaluation import *

# Load and preprocess the dataset
train_features, train_labels, test_features, test_labels = split_data()
X, y = split_data_into_features_and_labels()

# kmeans++
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(train_features, train_labels)
kmeans.plot_metrics()
test_predictions = kmeans.predict_and_map(test_features)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# softkmeans
soft_kmeans = SoftKMeans(k=3, max_iters=100, m=1.5)
soft_kmeans.fit(train_features, train_labels)
soft_kmeans.plot_metrics()
test_predictions = soft_kmeans.predict_and_map(test_features)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# pca
pca = PCA(n_components=2)
pca.fit(X)
pca.visualize(X, y)
pca = PCA(n_components=3)
pca.fit(X)
pca.visualize_3d(X, y)
pca.plot_explained_variance()

# nonlinear autoencoder
# Create and train the nonlinear autoencoder 2D
autoencoder_2d = NonlinearAutoencoder(input_dim=train_features.shape[1], hidden_dims=[64, 16, 2], output_dim=train_features.shape[1], learning_rate=1e-3, epochs=10000)
autoencoder_2d.train(train_features)
X_encoded = autoencoder_2d.encode(train_features)
autoencoder_2d.visualize(X_encoded, train_labels, n_components=2)
autoencoder_2d.plot_loss(n_components=2)
# Create and train the nonlinear autoencoder 3D
autoencoder_3d = NonlinearAutoencoder(input_dim=X.shape[1], hidden_dims=[64, 32, 8], output_dim=X.shape[1], learning_rate=1.5e-3, epochs=10000)
autoencoder_3d.train(X)
X_encoded_3d = autoencoder_3d.encode(X)
autoencoder_3d.visualize(X_encoded_3d, y, n_components=3)
autoencoder_3d.plot_loss(n_components=3)

# multi-layer perceptron
mlp = MultiLayerPerceptron([7, 64, 80, 32, 10], n_iter=1000, lr=3e-4, batch_size=32)
mlp.train(train_features, train_labels, test_features, test_labels)
mlp.plot_loss()
mlp.plot_weights_and_biases()
test_predictions = mlp.predict(test_features)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# pca and kmeans++
pca = PCA(n_components=2)
pca.fit(train_features)
train_pca = pca.transform(train_features)
test_pca = pca.transform(test_features)
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(train_pca, train_labels)
kmeans.plot_metrics()
test_predictions = kmeans.predict_and_map(test_pca)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# pca and softkmeans
pca = PCA(n_components=2)
pca.fit(train_features)
train_pca = pca.transform(train_features)
test_pca = pca.transform(test_features)
soft_kmeans = SoftKMeans(k=3, max_iters=100, m=1.5)
soft_kmeans.fit(train_pca, train_labels)
soft_kmeans.plot_metrics()
test_predictions = soft_kmeans.predict_and_map(test_pca)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# nonlinear autoencoder and kmeans++
autoencoder_2d = NonlinearAutoencoder(input_dim=train_features.shape[1], hidden_dims=[64, 16, 2], output_dim=train_features.shape[1], learning_rate=1e-3, epochs=10000)
autoencoder_2d.train(train_features)
train_encoded = autoencoder_2d.encode(train_features)
test_encoded = autoencoder_2d.encode(test_features)
autoencoder_2d.visualize(train_encoded, train_labels, n_components=2)
autoencoder_2d.plot_loss(n_components=2)
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(train_encoded, train_labels)
kmeans.plot_metrics()
test_predictions = kmeans.predict_and_map(test_encoded)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# nonlinear autoencoder and softkmeans
autoencoder_2d = NonlinearAutoencoder(input_dim=train_features.shape[1], hidden_dims=[64, 16, 2], output_dim=train_features.shape[1], learning_rate=1e-3, epochs=10000)
autoencoder_2d.train(train_features)
train_encoded = autoencoder_2d.encode(train_features)
test_encoded = autoencoder_2d.encode(test_features)
autoencoder_2d.visualize(train_encoded, train_labels, n_components=2)
autoencoder_2d.plot_loss(n_components=2)
soft_kmeans = SoftKMeans(k=3, max_iters=100, m=1.5)
soft_kmeans.fit(train_encoded, train_labels)
soft_kmeans.plot_metrics()
test_predictions = soft_kmeans.predict_and_map(test_encoded)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

# svm
svm = SVM(kernel='linear', C=1, k=3, n_iteration=30000, lr=0.05, early_stopping_patience=10, tol=1e-6)
svm.fit(train_features, train_labels)
test_pred = svm.get_predictions(test_features)
accuracy_score(test_labels, test_pred)
plot_confusion_matrix(test_labels, test_pred)
svm = SVM(kernel='gaussian', C=0.5, k=3, n_iteration=30000, lr=2, early_stopping_patience=100, tol=1e-12)
svm.fit(train_features, train_labels)
test_pred = svm.get_predictions(test_features)
accuracy_score(test_labels, test_pred)
plot_confusion_matrix(test_labels, test_pred)

# adaboost
model = AdaBoostMulticlass(n_estimators=10)
model.fit(train_features, train_labels, num_classes=3)
predictions = model.predict(test_features)
accuracy_score(test_labels, predictions)
plot_confusion_matrix(test_labels, predictions)


# binary problem
train_features, train_labels, test_features, test_labels = binary_dataset_split(1)

mlp = MultiLayerPerceptron([7, 64, 80, 10], n_iter=400, lr=5e-4, batch_size=32)
mlp.train(train_features, train_labels, test_features, test_labels)
mlp.plot_loss()
mlp.plot_weights_and_biases()
test_predictions = mlp.predict(test_features)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

svm = SVMBinary(kernel='linear', C=0.5, n_iteration=10000, lr=1e-6)
svm.fit(train_features, train_labels)
predictions = svm.get_predictions(test_features)
accuracy_score(test_labels, predictions)
plot_confusion_matrix(test_labels, predictions)
svm = SVMBinary(kernel='gaussian', C=0.1, n_iteration=10000, lr=1.5e-5)
svm.fit(train_features, train_labels)
predictions = svm.get_predictions(test_features)
accuracy_score(test_labels, predictions)
plot_confusion_matrix(test_labels, predictions)

model = AdaBoostMulticlass(n_estimators=10)
model.fit(train_features, train_labels, num_classes=2)
predictions = model.predict(test_features)
accuracy_score(test_labels, predictions)
plot_confusion_matrix(test_labels, predictions)