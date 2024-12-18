import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "True labels and predicted labels must have the same length"
    # compute the number of correct predictions
    correct = sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
    accuracy = correct / len(y_true)
    print(f"Accuracy: {accuracy}")

def compute_confusion_matrix(y_true, y_pred, num_classes):
    # initialize the confusion matrix
    assert len(y_true) == len(y_pred), "True labels and predicted labels must have the same length"
    cm = np.zeros((num_classes, num_classes), dtype=int)
    # fill the confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label-1, pred_label-1] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", cmap=plt.cm.Blues):
    # automatically determine the number of classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)
    # compute the confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    # plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()