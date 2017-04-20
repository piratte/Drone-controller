import sys
from pprint import pprint

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet, LinearRegression, MultiTaskLasso
from scipy.spatial.distance import sqeuclidean

JOINING_OFFSET = 1

if __name__ == '__main__':
    joined_data = np.load("joined_data.npy")
    new_navdata = np.load("new_navdata.npy")

    # training the actual flight model (state + command -> next state)

    X = joined_data[:-JOINING_OFFSET, 1:]
    y = new_navdata[JOINING_OFFSET:, 1:]

    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:int(X.shape[0]/6)], indices[int(X.shape[0]/6):]
    train_X, test_X, = X[training_idx, :], X[test_idx, :],
    train_y, test_y = y[training_idx, :], y[test_idx, :]

    for method in [LinearRegression, MultiTaskLasso, MultiTaskElasticNet]:
        try:
            clf = method(n_jobs=-1)
        except TypeError:
            clf = method()
        predictions = clf.fit(train_X, train_y).predict(X=test_X)

        score = list(map(lambda x: sqeuclidean(x[0], x[1]), zip(predictions, test_y)))
        max_ind = np.argmax(score)
        #print(predictions[max_ind, :])
        #print(test_y[max_ind])

        print(str(method.__name__), np.mean(list(score)))
