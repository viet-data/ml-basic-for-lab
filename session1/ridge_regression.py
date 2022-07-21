import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

PATH = "dataset/death_rate.txt"

#load data
def input(path):
    with open(path) as f:
        keys = list(f.readline().split())[1:];
        data = []
        for row in f:
            record = row.split()[1:]
            data.append(list(map(float, record)))
        df = pd.DataFrame(data=data, columns=keys)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

#normalize features to [0, 1] range by using min max algorithm
#adding feature x_0 = 1 to find bias w_0
def normalize_and_add_ones(data):
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    data = (data - min_)/(max_-min_)
    data.insert(0, "x0", np.ones(data.shape[0]))
    return data

class RidgeRegression():
    
    def __init__(self):
        self.w = np.array([])
        return
    
    #train model
    def fit(self, X_train, y_train, LAMBDA):
        assert len(X_train.shape) == 2 and y_train.shape[0] == X_train.shape[0]
        self.w = np.linalg.inv((X_train.T).dot(X_train)+
                             LAMBDA*np.identity(X_train.shape[1])).dot((X_train.T).dot(y_train))
        return self.w
    
    #train model by using gradient descent method
    def fit_gradient_descent(self, X_train, y_train, LAMBDA, learning_rate, max_num_epochs, batch_size=128):
        self.w = np.random.randn(X_train.shape[1])
        last_loss = self.compute_RSS(y_new=y_train, y_predicted=self.predict(X_new=X_train))
        
        for epoch in range(max_num_epochs):
            marks = list(range(len(X_train)))
            np.random.shuffle(marks)
            X_train = X_train.iloc[marks, :]
            y_train = y_train[marks]

            total_minibatch = int(np.ceil(len(y_train)/batch_size))
            
            for i in range(total_minibatch):
                index = i*batch_size
                X_train_sub = X_train[index: index+batch_size]
                y_train_sub = y_train[index: index+batch_size]
                grad = (self.w.dot(X_train_sub.T)-y_train_sub).dot(X_train_sub) + LAMBDA*self.w
                self.w = self.w-learning_rate*grad
                
            new_loss = self.compute_RSS(y_new=y_train, y_predicted=self.predict(X_train))
            if (np.abs(new_loss-last_loss) <= 1e-10):
                break
            last_loss=new_loss
        return self.w
        
    #Calculate outputs for given inputs
    def predict(self, X_new):
        assert self.w.shape[0] == X_new.shape[1]
        return X_new.dot(self.w.T)
    
    #Calculate the residual sum of squares
    def compute_RSS(self,  y_new, y_predicted):
        assert y_new.shape[0] == y_predicted.shape[0]
        return np.sum((y_new - y_predicted)**2)/(y_new.shape[0])
    
    #Find the best value of lambda by using k-fold method
    def get_the_best_lambda(self, X_train, y_train):
        best_LAMBDA, minimum_RSS = self.range_scan(best_LAMBDA=0, minimum_RSS=10000**2, LAMBDA_values=range(50));
        
        LAMBDA_values = [k*1./1000 for k in range(max(0, (best_LAMBDA-1)*1000), (best_LAMBDA+1)*1000, 1)]
        
        best_LAMBDA, minimum_RSS = self.range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)
        
        return best_LAMBDA
        
    #find the lambda value that gives smallest RSS
    def range_scan(self, best_LAMBDA, minimum_RSS, LAMBDA_values):
        for current_LAMBDA in LAMBDA_values:
            aver_RSS = self.cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
            if aver_RSS < minimum_RSS:
                minimum_RSS = aver_RSS
                best_LAMBDA = current_LAMBDA
        return best_LAMBDA, minimum_RSS
    
    #cross validation
    def cross_validation(self, num_folds, LAMBDA):
        rows_id = np.array(range(X_train.shape[0]))
        valid_ids = np.split(rows_id[:len(rows_id) - len(rows_id)%num_folds], num_folds)
        valid_ids[-1] = np.append(valid_ids[-1], rows_id[len(rows_id)-len(rows_id)%num_folds:])
        size_valid = int(len(rows_id) / num_folds)
        train_ids = [np.concatenate((rows_id[0:(i-1)*size_valid], rows_id[i*size_valid:]), axis=0) for i in range(1,  num_folds+1)]
        aver_RSS = 0
        
        for fold in range(num_folds):
            valid_part = {"X":X_train.iloc[valid_ids[fold], :], "Y":y_train[valid_ids[fold]]}
            train_part = {"X": X_train.iloc[train_ids[fold], :], "Y": y_train[train_ids[fold]]}
            model = RidgeRegression()
            model.fit(X_train=train_part["X"], y_train=train_part["Y"], LAMBDA=LAMBDA)
            aver_RSS += model.compute_RSS(y_new=valid_part["Y"], y_predicted=model.predict(X_new=valid_part["X"]))
        
        return aver_RSS/num_folds
    
    
if __name__ == "__main__":
    X, y = input(PATH)
    X = normalize_and_add_ones(data=X)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]
    
    model = RidgeRegression()
    
    #lambda = 0
    model.fit(X_train=X_train, y_train=y_train, LAMBDA=0)
    y_predicted = model.predict(X_new=X_test)
    print("RSS (Lambda=0):", model.compute_RSS(y_new=y_test, y_predicted=y_predicted))

    #determine the best value of lambda
    best_LAMBDA = model.get_the_best_lambda(X_train=X_train, y_train=y_train)
    print("Best LAMBDA:", best_LAMBDA)
    model.fit(X_train=X_train, y_train=y_train, LAMBDA=best_LAMBDA)
    y_predicted = model.predict(X_new=X_test)
    print("RSS of the best LAMBDA:", model.compute_RSS(y_new=y_test, y_predicted=y_predicted))
    
    
    
    