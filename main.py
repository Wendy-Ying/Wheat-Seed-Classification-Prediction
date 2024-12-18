from load import *
from kmeans import *
from softkmeans import *
from pca import *
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