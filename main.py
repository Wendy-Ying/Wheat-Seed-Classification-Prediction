from load import *
from kmeans import *
from softkmeans import *
from pca import *
from nonlinear import *
from evaluation import *

# Load and preprocess the dataset
train_features, train_labels, test_features, test_labels = split_data(load_data())

# # kmeans++
# kmeans = KMeans(k=3, max_iters=100)
# kmeans.fit(train_features, train_labels)
# kmeans.plot_metrics()
# test_predictions = kmeans.predict_and_map(test_features)
# accuracy_score(test_labels, test_predictions)
# plot_confusion_matrix(test_labels, test_predictions)

# # softkmeans
# soft_kmeans = SoftKMeans(k=3, max_iters=100, m=1.5)
# soft_kmeans.fit(train_features, train_labels)
# soft_kmeans.plot_metrics()
# test_predictions = soft_kmeans.predict_and_map(test_features)
# accuracy_score(test_labels, test_predictions)
# plot_confusion_matrix(test_labels, test_predictions)

# # pca
# X, y = split_data_into_features_and_labels(load_data())
# pca = PCA(n_components=2)
# pca.fit(X)
# pca.visualize(X, y)
# pca = PCA(n_components=3)
# pca.fit(X)
# pca.visualize_3d(X, y)
# pca.plot_explained_variance()

# # nonlinear autoencoder
# X, y = split_data_into_features_and_labels(load_data())
# # Create and train the nonlinear autoencoder 2D
# autoencoder_2d = NonlinearAutoencoder(input_dim=X.shape[1], hidden_dims=[64, 16, 2], output_dim=X.shape[1], learning_rate=1e-3, epochs=10000)
# autoencoder_2d.train(X)
# X_encoded = autoencoder_2d.encode(X)
# autoencoder_2d.visualize(X_encoded, y, n_components=2)
# autoencoder_2d.plot_loss(n_components=2)
# # Create and train the nonlinear autoencoder 3D
# autoencoder_3d = NonlinearAutoencoder(input_dim=X.shape[1], hidden_dims=[64, 32, 8], output_dim=X.shape[1], learning_rate=1.5e-3, epochs=10000)
# autoencoder_3d.train(X)
# X_encoded_3d = autoencoder_3d.encode(X)
# autoencoder_3d.visualize(X_encoded_3d, y, n_components=3)
# autoencoder_3d.plot_loss(n_components=3)

