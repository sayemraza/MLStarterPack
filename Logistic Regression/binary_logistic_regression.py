from math import exp
def sigmoid(x):
    return 1 / (1 + exp(-x))

from math import log
def CrossEntropy(yHat, y):
    if y == 1:
        return -log(yHat)
    else:
        return -log(1 - yHat)

def get_weights(W,X,yHat,y,learning_rate):
    gradient = np.dot(X.T,(yHat - y))
    avg_gradient = np.array(gradient)/X.shape[0]
    avg_gradient = avg_gradient * learning_rate
    W = W-avg_gradient
    return W

def get_loss(W,X,y):
    z = np.dot(X, W)
    yHat = np.array([sigmoid(each) for each in z], dtype = np.float64)
    ce = []
    for i in range(len(yHat)):
        ce.append(CrossEntropy(yHat[i],y[i]))
    loss = sum(ce)/yHat.shape[0]
    return loss, yHat

def train(epochs,n_features,X_train,y_train,learning_rate=0.1):
    W = np.random.normal(size=n_features)
    for epoch in range(epochs):
        loss,yHat = get_loss(W,X_train,y_train)
        W = get_weights(W,X_train,yHat,y_train,learning_rate = learning_rate)
    return W,loss
