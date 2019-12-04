def normalize(X_train):
    for x in X_train.T:
        mean = np.mean(x)
        range = np.amax(x) - np.amin(x)
        x -= mean
        x /= range
    return X_train
def least_aquare(X_train, y_train,yhat, W):
    sq_error = (yhat - y_train)**2
    return 1.0/(2*y_train.shape[0]) * sq_error.sum()

def gradient_descent(X_train, y_train, yhat, W, learning_rate):
    gradient = np.dot(X_train.T,  (yhat-y_train))
    avg_gradient = gradient/X_train.shape[0]
    avg_gradient *= learning_rate
    weights = W - avg_gradient
    return weights
def train(epochs, X_train, y_train, n_features, learning_rate=0.01):
    #Initialize random weights
    W = np.random.normal(size=n_features)
    X_train = normalize(X_train)
    for epoch in range(epochs):
        #Make the predictions
        yhat = np.dot(X_train, W.T)
        #compute loss
        loss = least_aquare(X_train, y_train,yhat, W)
        #update weights using gradient descent
        W = gradient_descent(X_train, y_train, yhat, W, learning_rate)
    return loss, W
