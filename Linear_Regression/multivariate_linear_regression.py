import numpy as np
class linear_regression:
    def __init__(self, X_train, y_train, n_features):
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = n_features
    def normalize(self):
        for x in self.X_train.T:
            mean = np.mean(x)
            range = np.amax(x) - np.amin(x)
            x -= mean
            x /= range
        return self.X_train
    def least_aquare(self,yhat, W):
        sq_error = (yhat - self.y_train)**2
        return 1.0/(2*self.y_train.shape[0]) * sq_error.sum()

    def gradient_descent(self, yhat, W, learning_rate):
        gradient = np.dot(self.X_train.T,  (yhat-self.y_train))
        avg_gradient = gradient/self.X_train.shape[0]
        avg_gradient *= learning_rate
        weights = W - avg_gradient
        return weights
    def train(self, epochs, learning_rate=0.01):
        #Initialize random weights
        W = np.random.normal(size=self.n_features)
        self.X_train = self.normalize(self.X_train)
        for epoch in range(epochs):
            #Make the predictions
            yhat = np.dot(self.X_train, W.T)
            #compute loss
            loss = self.least_aquare(self.X_train, self.y_train,yhat, W)
            #update weights using gradient descent
            W = self.gradient_descent(self.X_train, self.y_train, yhat, W, learning_rate)
        return loss, W
