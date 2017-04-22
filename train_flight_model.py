import sys
from pprint import pprint

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet, LinearRegression, MultiTaskLasso
from scipy.spatial.distance import sqeuclidean

JOINING_OFFSET = 1
#WINDOW_LENGTH = 10
#NAVDATA_OFFSET_MILIS = 10

if __name__ == '__main__':
    scores = []

    for WINDOW_LENGTH, NAVDATA_OFFSET_MILIS in [(10,10), (15,10), (15,15), (20,10), (20,15)]:
        print(WINDOW_LENGTH, NAVDATA_OFFSET_MILIS)
        joined_data = np.load("joined_data_long_minimal_%d_%d.npy" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS))
        #joined_data = np.load("joined_data_minimal_%d_%d.npy" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS))
        new_navdata = np.load("new_navdata_long_minimal_%d_%d.npy" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS))
        #new_navdata = np.load("new_navdata_minimal_%d_%d.npy" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS))

        # training the actual flight model (state + command -> next state)

        # strip the data of timestamp and "state"  columns
        X = joined_data[:-JOINING_OFFSET, 2:]
        y = new_navdata[JOINING_OFFSET:, 2:]

        # split the dataset
        np.random.seed(5)
        indices = np.random.permutation(X.shape[0])
        training_idx, test_idx = indices[:int(X.shape[0]/6)], indices[int(X.shape[0]/6):]
        train_X, test_X, = X[training_idx, :], X[test_idx, :],
        train_y, test_y = y[training_idx, :], y[test_idx, :]

        # try three regressors
        for method in [LinearRegression, MultiTaskLasso, MultiTaskElasticNet]:
            try:
                clf = method(n_jobs=-1)
            except TypeError:
                clf = method(max_iter=8000)

            predictions = clf.fit(train_X, train_y).predict(X=test_X)
            score = list(map(lambda x: sqeuclidean(x[0], x[1]), zip(predictions, test_y)))

            max_ind = np.argmax(score)
            print(predictions[max_ind, :])
            print(test_y[max_ind])

            scores.append(np.mean(list(score)))
            print(str(method.__name__), scores[-1])

    print(np.min(scores))
