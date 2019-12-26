import numpy as np

class binary_logistic_regression:

    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train.reshape(y_train.shape[0],1)
        self.n_features = self.X_train.shape[1]
    
    def next_batch(self,X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])


    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def CrossEntropy(self,yHat, y):
        if y == 1:
            return -np.log(yHat)
        else:
            return -np.log(1 - yHat)

    def update_weights(self,yHat):
        gradient = np.dot(self.X_train.T,(yHat - self.y_train))
        avg_gradient = np.array(gradient)/self.X_train.shape[0]
        avg_gradient = avg_gradient * self.learning_rate
        self.W = self.W-avg_gradient


    def fit(self,learning_rate=0.01, epochs = 10000):
        self.learning_rate = learning_rate
        self.W = np.random.normal(size=(self.n_features,1))
        for epoch in range(epochs):
            z = np.dot(self.X_train, self.W)
            yHat = np.array([self.sigmoid(each) for each in z], dtype = np.float64)
            ce = []
            for i in range(len(yHat)):
                ce.append(self.CrossEntropy(yHat[i],self.y_train[i]))
            self.loss = sum(ce)/yHat.shape[0][0]
            self.update_weights(yHat)
            if epoch%100 == 0:
                print('Epoch: ', epoch, 'Loss: ', self.loss)
    
    def predict(self,X_test):
        z = np.dot(self.X_train, self.W)
        yHat = np.array([self.sigmoid(each) for each in z], dtype = np.float64)
        yHat[yHat>0.5]=1
        yHat[yHat<0.5]=0
        return yHat
