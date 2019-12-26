import numpy as np
class linear_regression:

    def __init__(self, X_train, y_train, n_features):
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = n_features
        
    def least_aquare(self,yhat):
        sq_error = (yhat - self.y_train)**2
        return 1.0/(2*self.y_train.shape[0]) * sq_error.sum()

    def gradient_descent(self, yhat):
        gradient = np.dot(self.X_train.T,  (yhat-self.y_train))
        avg_gradient = gradient/self.X_train.shape[0]
        avg_gradient *= self.learning_rate
        self.W = self.W - avg_gradient

    def fit(self, epochs, learning_rate=0.01):
        self.learning_rate = learning_rate
        #Initialize random weights
        self.W = np.random.normal(size=self.n_features)
        for epoch in range(epochs):
            #Make the predictions
            yhat = np.dot(self.X_train, self.W.T)
            #compute loss
            loss = self.least_aquare(yhat)
            #update weights using gradient descent
            self.gradient_descent(yhat)
    
    def predict(self,X_test):
        yhat = np.dot(self.X_train, self.W.T)
        return yhat
