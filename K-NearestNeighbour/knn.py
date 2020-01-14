import numpy as np

class knn:

    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train  = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.k_iterations = self.X_train[0]

    def euclidean_distance(self,a,b):
        return np.linalg.norm(a-b)
    
    def get_class_prediction(self,distance_test,k):
        y_pred = []
        for k1,v1 in distance_test.items():
            vals = list(v1.values())
            vals.sort()
            vals = vals[:k+1]
            classes = []
            for n,value in v1.items():
                if value in vals:
                    classes.append(self.y_train[n])
            y_pred.append(max(set(classes), key=classes.count))
        return y_pred
    
    def get_loss(self,y_pred):
        incorrect_predictions = np.where(y_pred != self.y_test)[0].shape[0]
        loss = incorrect_predictions/self.y_test.shape[0]
        return loss

    def fit(self):
        self.loss = []
        y_pred = []
        distance_test = {}
        for i,a in enumerate(self.X_test):
            distance = {}
            for j,b in enumerate(self.X_train):
                dis = self.euclidean_distance(a,b)
                distance[j] = dis
            distance_test[i]=distance
        for k in range(self.X_train.shape[0]):
            y_pred = self.get_class_prediction(distance_test,k)
            self.loss.append(self.get_loss(np.array(y_pred)))
        self.k_optimum = self.loss.index(min(self.loss))+1
        
    def predict(self,X_pred):
        y_pred = []
        distance_test = {}
        for i,a in enumerate(self.X_test):
            distance = {}
            for j,b in enumerate(self.X_train):
                dis = self.euclidean_distance(a,b)
                distance[j] = dis
            distance_test[i]=distance
        y_pred = self.get_class_prediction(distance_test,self.k_optimum)
        return y_pred
