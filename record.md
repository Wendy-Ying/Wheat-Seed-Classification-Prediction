# 3 class problem

## kmeans++

Converged at iteration 25
Accuracy: 0.9523809523809523

## softkmeans

Converged at iteration 30
Accuracy: 0.9047619047619048

## mlp

Epoch 1/1000, Train Loss: 25.9276, Test Loss: 25.5430
Epoch 40/1000, Train Loss: 0.7140, Test Loss: 0.6661
Epoch 80/1000, Train Loss: 0.4441, Test Loss: 0.4046
Epoch 120/1000, Train Loss: 0.3791, Test Loss: 0.3347
Epoch 160/1000, Train Loss: 0.3408, Test Loss: 0.3162
Epoch 200/1000, Train Loss: 0.3044, Test Loss: 0.2764
Epoch 240/1000, Train Loss: 0.2856, Test Loss: 0.2622
Epoch 280/1000, Train Loss: 0.2730, Test Loss: 0.2635
Epoch 320/1000, Train Loss: 0.2612, Test Loss: 0.2430
Epoch 360/1000, Train Loss: 0.2565, Test Loss: 0.2318
Epoch 400/1000, Train Loss: 0.2470, Test Loss: 0.2285
Epoch 440/1000, Train Loss: 0.2586, Test Loss: 0.2359
Epoch 480/1000, Train Loss: 0.2367, Test Loss: 0.2371
Epoch 520/1000, Train Loss: 0.2333, Test Loss: 0.2216
Epoch 560/1000, Train Loss: 0.2216, Test Loss: 0.2187
Epoch 600/1000, Train Loss: 0.2195, Test Loss: 0.2276
Epoch 640/1000, Train Loss: 0.2126, Test Loss: 0.2236
Epoch 680/1000, Train Loss: 0.2077, Test Loss: 0.2207
Epoch 720/1000, Train Loss: 0.2029, Test Loss: 0.2176
Epoch 760/1000, Train Loss: 0.1994, Test Loss: 0.2170
Epoch 800/1000, Train Loss: 0.1946, Test Loss: 0.2154
Epoch 840/1000, Train Loss: 0.1921, Test Loss: 0.2146
Epoch 880/1000, Train Loss: 0.1890, Test Loss: 0.2202
Epoch 920/1000, Train Loss: 0.1840, Test Loss: 0.2092
Epoch 960/1000, Train Loss: 0.1823, Test Loss: 0.2157
Epoch 1000/1000, Train Loss: 0.1776, Test Loss: 0.2115
Accuracy: 0.9523809523809523

## pca, nonlinear autoencoder and kmeans++, softkmeans

### pca and kmeans++
Converged at iteration 40
Accuracy: 0.9285714285714286

### pca and softkmeans
Converged at iteration 38
Accuracy: 0.9285714285714286

### nonlinear autoencoder and kmeans++
Converged at iteration 35
Accuracy: 0.9523809523809523

### nonlinear autoencoder and softkmeans
Converged at iteration 52
Accuracy: 0.9285714285714286

## svm

### linear kernal
Early stopping at iteration 6985
Early stopping at iteration 11750
Early stopping at iteration 18295
Accuracy: 0.9047619047619048

### gaussian kernal
Early stopping at iteration 2496
Early stopping at iteration 2590
Early stopping at iteration 3102
Accuracy: 0.9523809523809523

## adaptive boosting
Accuracy: 0.9285714285714286


# binary problem

## Removing class: 3
mlp: 1.0
svm linear: 0.9285714285714286
svm gaussian: 0.9642857142857143
adaboost: 0.9642857142857143

## Removing class: 2
mlp: 0.9285714285714286
svm linear: 0.9285714285714286
svm gaussian: 0.9285714285714286
adaboost: 0.9642857142857143

## Removing class: 1
mlp: 1.0
svm linear: 1.0
svm gaussian: 1.0
adaboost: 1.0