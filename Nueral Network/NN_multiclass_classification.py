class NeuralNetwork:
    #Network Architecture
"""
L1 -> type = input, size = n_features
L2 -> type = hidden, size = 5
L3 -> type = hidden, size =6
L4 -> type = output, size = n_classes
"""
    
    def __init__(self, X_train, Y_train):
        self.input = X_train
        self.y  = y_train
        self.n_features = X_train.shape[1]
        self.n_classes = Y_train.shape[1]
        self.W1 = np.random.randn(self.n_features,5)
        self.W2 = np.random.randn(5,6)
        self.W3 = np.random.randn(6,self.n_classes)
        self.B1 = np.zeros((1,5))
        self.B2 = np.zeros((1,6))
        self.B3 = np.zeros((1,self.n_classes))
    
    def ReLU(z):
        for arr in z:
            r = []
            for val in arr:
                if val>0:
                    r.append(val)
                else:
                    r.append(0)
    def reluDerivative(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def softmax(z):
        result = []
        for each in z:
            exp_sum = sum([np.exp(i) for i in each])
            exp = [np.exp(i) for i in each]
            s_max = exp / exp_sum
            result.append(s_max)
        return np.array(result)
    
    def categorical_cross_entropy(y,pHat):
        log_sum = []
        for i in range(Y.shape[0]):
            log_sum.append(sum([a*np.log(b) for a,b in zip(Y[i],pHat[i])]))
        loss = - sum(log_sum)/Y.shape[1]
        return loss
    
    def feed_forward(self):
        z1 = np.dot(self.X_train,self.W1)+self.B1
        z2 = np.dot(self.ReLU(z1),self.W2)+self.B2
        z3 = np.dot(self.ReLU(z2),self.W3)+self.B3
        phat = self.softmax(z3)
        loss = self.categorical_cross_entropy(self.Y_train,phat)
        params = {'z1':z1,'z2':z2,'z3':z3,'phat':phat,'loss':loss}
        return params
    
    def back_prop(self, params):
      pass
