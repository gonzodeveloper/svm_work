from sklearn.svm import  SVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, RBF
from sklearn.preprocessing import StandardScaler
from point_generator import *
import pandas as pd

def simulation(n, dim, dist, data_class, point_param, model_type, param, kern, runs, train_ratio=0.8):
    '''
    Run a series of simulations with an SVC. Given a set of labeled points create a train and test split.
    Record the error on testing classification as well as other parameters given to the SVC
    :param n: number of points, total
    :param runs: number of runs to test
    :param constant: C value for SVC
    :param margin: minimum margin of separation for points
    :param train_ratio: ratio of points used for training data points float between 0 and 1
    :param d: dimensionality of points
    :return: dataframe with recorded data
    '''
    # binary or multiclass data
    binary = True if data_class == 'binary' else False
    multi = 'one_vs_rest' if binary else 'one_vs_one'
    # if using gpc we need a kernel object
    if model_type == 'gpc':
        if kern == 'rbf':
            kern = RBF()
        elif kern == 'linear':
            kern = DotProduct()
    else:
        if kern == 'rbf':
            parameters = {'kernel': ('rbf', 'rbf'), 'gamma':[1e-3, 1e3]}
        elif kern == 'linear':
            parameters = {'kernel': ('linear', 'linear'), 'gamma':['auto','auto']}

    all_data = []
    for i in range(runs):
        # Get data according to given distribution and parameters
        if dist == 'linear':
            data = linear_labeled_points(n, dim, gamma=point_param)
        elif dist == 'islands':
            data = island_labeled_points(n, dim, gamma=point_param)
        elif dist == 'gauss':
            data = gaussian_labeled_points(n, dim, gamma=point_param, binary=binary)
        elif dist == 'random_walk':
            data = random_walks(n, p=point_param)
        else:
            data = None
        # train test split according to ratio
        train_dat, test_dat = train_test_split(data, train_size=train_ratio, test_size=(1-train_ratio))
        # Separate train points from labels
        train_points = [x[0] for x in train_dat]
        train_labels = [x[1] for x in train_dat]

        # Separate test points from their labels
        test_points = [x[0] for x in test_dat]
        test_labels = [x[1] for x in test_dat]

        # Get a model specified, fit to data, score for error, mark error as -1 if fails
        try:
            if model_type == 'svc':
                model = SVC(kernel=kern, C=param)
                clf = GridSearchCV(model, parameters)
                clf.fit(train_points, train_labels)
                model_error = 1 - clf.score(test_points, test_labels)
            elif model_type == 'nusvc':
                model = NuSVC(kernel=kern, nu=param)
                clf = GridSearchCV(model, parameters)
                clf.fit(train_points, train_labels)
                model_error = 1 - clf.score(test_points, test_labels)
            elif model_type == 'gpc':
                model = GaussianProcessClassifier(kernel=kern, multi_class=multi)
                model.fit(train_points, train_labels)
                model_error = 1 - model.score(test_points, test_labels)
            else:
                model = None
                model_error = -1
        except ValueError:
            model_error = -1

        all_data.append([n, train_ratio, point_param, param, model_error])

    df = pd.DataFrame(all_data, columns=['n', 'train_ratio', 'point_param', 'machine_param', 'test_error'])
    return df


if __name__ == "__main__":
    pass
