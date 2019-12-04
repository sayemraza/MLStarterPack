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

def get_weights(W,pHat,y,learning_rate):
    gradient = np.dot(X.T,(pHat - y))
    avg_gradient = np.array(gradient)/Y.shape[0]
    avg_gradient = avg_gradient * learning_rate
    W = W-avg_gradient.T
    return W

def train(epochs,n_class,n_features,X_train,Y_train,learning_rate=0.1):
  W = np.random.rand(n_class, n_features)
  for i in rangex(epochs):
      z = np.dot(X_train, W.T)
      pHat = softmax(z)
      yHat = np.array([np.argmax(i) for i in pHat])
      W = get_weights(W,pHat,Y_train,learning_rate)
