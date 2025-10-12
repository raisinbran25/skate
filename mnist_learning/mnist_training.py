import numpy as np

def initialize(nIn, nLay1, nLay2, nOut):
    w1 = np.random.randn(nLay1, nIn) * np.sqrt(1/nIn)

    w2 = np.random.randn(nLay2, nLay1) * np.sqrt(1/nLay2)

    w3 = np.random.randn(nOut, nLay2) * np.sqrt(1/nLay2)

    return w1, w2, w3

def softmax(x, derivative = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)
    

def push(filename, w1, w2, w3):
    np.savez(filename, w1=w1, w2=w2, w3=w3)

def pull(filename):
    d = np.load(filename + ".npz")
    return d["w1"], d["w2"], d["w3"]

def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

# Begin here
w1, w2, w3 = initialize(784, 64, 64, 10)

# import mnist database
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# X shape: (70000, 784), y shape: (70000,), y dtype is str — cast if needed
X = X.astype('float32') / 255.0
y = y.astype('int64')

# Split like Keras:
x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# If you want 28x28 images:
x_train_img = x_train.reshape(-1, 28, 28)
x_test_img  = x_test.reshape(-1, 28, 28)

# baseline version
push("mnistv1", w1, w2, w3)


# actual training below

epochs = 25
for j in range(epochs):

    # randomize order
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    loss = 0

    for i in range(0, 60000):
        
        # set ouput neuron
        yVec = np.zeros(10) # 10 x 1
        for k in range(10):
            if y_train[i] == k:
                yVec[k] = 0.99
            else:
                yVec[k] = 0.01

        # set input neurons
        a0 = x_train[i] # 784 x 1

        #forward prop
        z1 = w1 @ a0
        a1 = sigmoid(z1)

        z2 = w2 @ a1
        a2 = sigmoid(z2)

        z3 = w3 @ a2
        a3 = softmax(z3)

        #back prop 
        error = 2 * (a3 - yVec) / a3.shape[0] * softmax(z3, derivative=True)
        dw3 = np.outer(error, a2)

        error = w3.T @ error * sigmoid(z2, derivative=True)
        dw2 = np.outer(error, a1)

        error = w2.T @ error * sigmoid(z1, derivative=True)
        dw1 = np.outer(error, a0)
        
        #update weights and biases
        lr = 0.01
        w3 -= lr * dw3

        w2 -= lr * dw2

        w1 -= lr * dw1

    print("epoch " + str(j + 1) + " loss: " + str(loss))


#push trained model
push("mnistv2", w1, w2, w3)