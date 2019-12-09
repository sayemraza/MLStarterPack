class NeuralNetwork:
    #Network Architecture
    """
    L1 -> type = input, size = n_features
    L2 -> type = hidden, size = 5
    L3 -> type = hidden, size =6
    L4 -> type = output, size = n_classes
    """
    
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train  = Y_train
        self.n_features = self.X_train.shape[1]
        self.n_classes = self.Y_train.shape[1]
    
    def initilize_weights(self):
        self.W1 = np.random.normal(loc=0, scale=np.sqrt(2/self.n_features+self.n_classes), size=(self.n_features,5))
        self.W2 = np.random.normal(loc=0, scale=np.sqrt(2/self.n_features+self.n_classes), size=(5,6))
        self.W3 = np.random.normal(loc=0, scale=np.sqrt(2/self.n_features+self.n_classes), size=(6,self.n_classes))
        self.B1 = np.zeros((1,5))
        self.B2 = np.zeros((1,6))
        self.B3 = np.zeros((1,self.n_classes))
    
    def ReLU(self,z):
        relu = []
        for arr in z:
            r = []
            for val in arr:
                if val>0:
                    r.append(val)
                else:
                    r.append(0)
            relu.append(r)
        return np.array(relu)
    def reluDerivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def softmax(self,z):
        result = []
        for each in z:
            exp_sum = sum([np.exp(i) for i in each])
            exp = [np.exp(i) for i in each]
            s_max = exp / exp_sum
            result.append(s_max)
        return np.array(result)
    def softmaxDerivative(self,s):
        SM = s.reshape((-1,1))
        jac = np.diagflat(s) - np.dot(SM, SM.T)
        return list(jac.diagonal())
    
    def gradient_clipping(self,x,threshold):
        if np.linalg.norm(x) > threshold:
            return (x * threshold/norm(x))
        else:
            return x
    
    def categorical_cross_entropy(self,Y,phat):
        log_sum = []
        for i in range(Y.shape[0]):
            log_sum.append(sum([a*np.log(max(b, 1e-9)) for a,b in zip(Y[i],phat[i])]))
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
    
    def back_prop(self, params,learning_rate):
        softmax_gradient = []
        for each in params['phat']:
            softmax_gradient.append(self.softmaxDerivative(each))
        softmax_gradient = np.array(softmax_gradient)
        dz3 = np.dot(params['z2'].T, ((params['phat']-self.Y_train) * softmax_gradient))
        dz2 = np.dot(params['z1'].T,np.dot((params['phat']-self.Y_train) * softmax_gradient,self.W3.T)*self.reluDerivative(params['z2']))
        dz1 = np.dot(self.X_train.T,np.dot(np.dot((params['phat']-self.Y_train) * softmax_gradient,self.W3.T)*self.reluDerivative(params['z2']),self.W2.T))
        d_weights1 = dz1*learning_rate
        d_weights2 = dz2*learning_rate
        d_weights3 = dz3*learning_rate
        # update the weights with the derivative (slope) of the loss function
        self.W1 -= d_weights1
        self.W2 -= d_weights2
        self.W3 -= d_weights3
        
    def fit(self,epochs,learning_rate=0.01,n_iterations=3000):
        loss = []
        weights = []
        for epoch in range(epochs):
            self.initilize_weights()
            for n in range(n_iterations):
                params = self.feed_forward()
                self.back_prop(params,learning_rate)
                weights.append({'W1':self.W1,'W2':self.W2,'W3':self.W3})
                loss.append(params['loss'])
            print('Epoch: ',epoch,'Loss: ',loss[np.argmin(loss)])
            self.current_loss = loss[np.argmin(loss)]
        self.min_loss = loss[np.argmin(loss)]
        self.weights = weights[np.argmin(loss)]
    def predict(self,X_test):
        z1 = np.dot(X_test,self.W1)+self.B1
        z2 = np.dot(self.ReLU(z1),self.W2)+self.B2
        z3 = np.dot(self.ReLU(z2),self.W3)+self.B3
        phat = self.softmax(z3)
        print(phat)
        ypred = np.array([np.argmax(i) for i in phat], dtype = np.uint8)
        return ypred
        
