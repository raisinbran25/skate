import numpy as np
from sklearn.datasets import fetch_openml
import pandas

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# X shape: (70000, 784), y shape: (70000,), y dtype is str — cast if needed
X = X.astype('float32') / 255.0
y = y.astype('int64')

# Split like Keras:
x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Keep (x, y) pairing consistent when shuffling
perm = np.random.permutation(len(x_test))
x_test = x_test[perm]
y_test = y_test[perm]

# If you want 28x28 images:
x_train_img = x_train.reshape(-1, 28, 28)
x_test_img  = x_test.reshape(-1, 28, 28)

def pull(filename):
    load = np.load(filename + ".npz")
    
    w1 = load["w1"]

    w2 = load["w2"]

    w3 = load["w3"]

    return w1, w2, w3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

w1, w2, w3 = pull("mnistv1")
correct = 0

for i in range(0, 10000):

    # set input neurons
    a0 = x_test[i] # 784 x 1

    #forward prop
    z1 = w1 @ a0
    a1 = sigmoid(z1)

    z2 = w2 @ a1
    a2 = sigmoid(z2)

    z3 = w3 @ a2
    a3 = softmax(z3)

    # was the model correct?
    if y_test[i] == np.argmax(a3):
        correct += 1

    if i == 9999:
        print("model v1 got " + str(correct) + " correct.")



# now test the trained model

w1, w2, w3 = pull("mnistv2")
correct = 0

for i in range(0, 10000):

    # set input neurons
    a0 = x_test[i] # 784 x 1

    #forward prop
    z1 = w1 @ a0
    a1 = sigmoid(z1)

    z2 = w2 @ a1
    a2 = sigmoid(z2)

    z3 = w3 @ a2
    a3 = softmax(z3)

    # was the model correct?
    if y_test[i] == np.argmax(a3):
        correct += 1

    if i == 9999:
        print("model v2 got " + str(correct) + " correct.")