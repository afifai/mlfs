# import library
import numpy as np
from util import get_data
from datetime import datetime
from sortedcontainers import SortedList
import operator
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        # loop untuk semua data X baru
        for j, x in enumerate(X):
            sl = SortedList()
            # loop untuk data train
            for i, xt in enumerate(self.X):
                # hitung jarak
                dist = x - xt
                dist = dist.dot(dist)
                if len(sl) < self.k:
                    sl.add((dist, self.y[i]))
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add((dist, self.y[i]))
            # vote 
            votes = {}
            for e in sl:
                if e[1] in votes.keys():
                    votes[e[1]] += 1
                else:
                    votes[e[1]] = 1
            
            # cari nilai max
            max_votes_class, max_votes = max(votes.items(), key=operator.itemgetter(1))
            y[j] = max_votes_class
        return y
    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

if __name__ == '__main__':
    X, y = get_data(2000)
    Ntrain = 1000
    X_train, y_train = X[:Ntrain], y[:Ntrain]
    X_test, y_test = X[Ntrain:], y[Ntrain:]

    trainAcc = []
    testAcc = []
    for k in range(1,6):
        print("="*50)
        print("k :", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(X_train, y_train)
        print("Waktu training : ", datetime.now() - t0)

        t0 = datetime.now()
        train_acc = knn.score(X_train, y_train)
        trainAcc.append(train_acc)
        print("Akurasi Training :", train_acc)
        print("Waktu predict train : ", datetime.now() - t0)
        print("Ukuran data training : ", X_train.shape)

        t0 = datetime.now()
        test_acc = knn.score(X_test, y_test)
        testAcc.append(test_acc)
        print("Akurasi Testing :", test_acc)
        print("Waktu predict test : ", datetime.now() - t0)
        print("Ukuran data testing : ", X_test.shape)
        print("="*50)
    
    plt.plot(range(1,6), trainAcc, label='Akurasi Training')
    plt.plot(range(1,6), testAcc, label='Akurasi Testing')
    plt.legend()
    plt.show()
