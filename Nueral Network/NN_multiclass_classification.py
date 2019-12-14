import numpy as np
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
        self.history = {'loss':[], 'weights':[], 'gradients':[]}

    def initialize_weights(self):
        """
        Xavier's initialization
        """
        self.W1 = np.random.normal(loc=0, scale=np.sqrt(2)*np.sqrt(2/(self.n_features+self.n_classes)), size=(self.n_features,5))
        self.W2 = np.random.normal(loc=0, scale=np.sqrt(2)*np.sqrt(2/(self.n_features+self.n_classes)), size=(5,6))
        self.W3 = np.random.normal(loc=0, scale=np.sqrt(2/(self.n_features+self.n_classes)), size=(6,self.n_classes))
        self.B1 = np.zeros((1,5))
        self.B2 = np.zeros((1,6))
        self.B3 = np.zeros((1,self.n_classes))

    def next_batch(self,X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])
    
    def ReLU(self,z):
        z = np.maximum(z, 0)
        return z

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
    
    def categorical_cross_entropy(self,Y,phat):
        log_sum = []
        for i in range(Y.shape[0]):
            log_sum.append(sum([a*np.log(max(b,1e-9)) for a,b in zip(Y[i],phat[i])]))
        loss = - sum(log_sum)/Y.shape[1]
        return loss
    
    def feed_forward(self):
        z1 = np.dot(self.X_mini_batch,self.W1)+self.B1
        z2 = np.dot(self.ReLU(z1),self.W2)+self.B2
        z3 = np.dot(self.ReLU(z2),self.W3)+self.B3
        phat = self.softmax(z3)
        loss = self.categorical_cross_entropy(self.Y_mini_batch,phat)
        params = {'z1':z1,'z2':z2,'z3':z3,'phat':phat,'loss':loss}
        return params
    
    def back_prop(self, params):
        softmax_gradient = []
        for each in params['phat']:
            softmax_gradient.append(self.softmaxDerivative(each))
        softmax_gradient = np.array(softmax_gradient)
        dz3 = np.dot(params['z2'].T, ((params['phat']-self.Y_mini_batch) * softmax_gradient))
        dz2 = np.dot(params['z1'].T,np.dot((params['phat']-self.Y_mini_batch) * softmax_gradient,self.W3.T)*self.reluDerivative(params['z2']))
        dz1 = np.dot(self.X_mini_batch.T,np.dot(np.dot((params['phat']-self.Y_mini_batch) * softmax_gradient,self.W3.T)*self.reluDerivative(params['z2']),self.W2.T))
        self.d_weights1 = dz1*self.learning_rate
        self.d_weights2 = dz2*self.learning_rate
        self.d_weights3 = dz3*self.learning_rate
        self.history['gradients'].append((self.d_weights1,self.d_weights2,self.d_weights3))
        self.gradient_descent()

    def gradient_descent(self, momentum=0.1):
        # update the weights with the derivative (slope) of the loss function
        # Use momentum = 0.1
        try:
            dw1_dash = self.history['gradients'][-2][0]
            dw2_dash = self.history['gradients'][-2][1]
            dw3_dash = self.history['gradients'][-2][2]
        except:
            dw1_dash = 0
            dw2_dash = 0
            dw3_dash = 0

        self.W1 -= self.learning_rate*dw1_dash + self.d_weights1
        self.W2 -= self.learning_rate*dw2_dash + self.d_weights2
        self.W3 -= self.learning_rate*dw3_dash + self.d_weights3
        self.history['weights'].append((self.W1,self.W2,self.W3))
        
    def fit(self,epochs,learning_rate=0.01, batchSize = 32):
        self.learning_rate = learning_rate
        self.initialize_weights()
        for epoch in range(epochs):
            # self.initialize_weights()
            batch_loss = []
            for each in self.next_batch(self.X_train, self.Y_train, batchSize):
                self.X_mini_batch = each[0]
                self.Y_mini_batch = each[1]
                params = self.feed_forward()
                self.back_prop(params)
                batch_loss.append(params['loss'])
            self.history['loss'].append(np.mean(batch_loss))           
            print('Epoch: ',epoch+1, 'Loss: ',self.history['loss'][-1])

    def predict(self,X_test):
        z1 = np.dot(X_test,self.W1)+self.B1
        z2 = np.dot(self.ReLU(z1),self.W2)+self.B2
        z3 = np.dot(self.ReLU(z2),self.W3)+self.B3
        phat = self.softmax(z3)
        ypred = np.array([np.argmax(i) for i in phat], dtype = np.uint8)
        return ypred
