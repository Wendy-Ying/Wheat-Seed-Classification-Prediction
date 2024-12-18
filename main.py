from load import *
from kmeans import *
from evaluation import *

# Load and preprocess the dataset
train_features, train_labels, test_features, test_labels = split_data(load_data())

# kmeans++
kmeans = KMeans(k=3)
kmeans.fit(train_features, train_labels)
kmeans.plot_metrics()
test_predictions = kmeans.predict_and_map(test_features)
accuracy_score(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)

