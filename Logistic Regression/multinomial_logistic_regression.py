import numpy as np

class multinomial_logistic_regression:

    def __init__(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = self.X_train.shape[1]
        self.n_classes = self.y_train.shape[1]
        
    def softmax(self,z):
        result = []
        for each in z:
            exp_sum = sum([np.exp(i) for i in each])
            exp = [np.exp(i) for i in each]
            s_max = exp / exp_sum
            result.append(s_max)
        return np.array(result)

    def categorical_cross_entropy(self,y,pHat):
        log_sum = []
        for i in range(y.shape[0]):
            log_sum.append(sum([a*np.log(b) for a,b in zip(y[i],pHat[i])]))
        loss = - sum(log_sum)/y.shape[1]
        return loss

    def update_weights(self,W,pHat,y,learning_rate):
        gradient = np.dot(self.X_train.T,(pHat - y))
        avg_gradient = np.array(gradient)/self.y_train.shape[0]
        avg_gradient = avg_gradient * learning_rate
        W = W-avg_gradient
        return W

    def fit(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.W = np.random.normal(size = (self.n_features,self.n_classes))
        for i in range(epochs):
            z = np.dot(self.X_train, self.W)
            pHat = self.softmax(z)
            self.loss = self.categorical_cross_entropy(self.y_train,pHat)
            if i%100==0:
                print(self.loss)
            self.W = self.update_weights(self.W,pHat,self.y_train, self.learning_rate)
    
    def predict(X_test):
        z = np.dot(self.X_train, self.W)
        pHat = self.softmax(z)
        yHat = np.array([np.argmax(i) for i in pHat])
        return yHat
