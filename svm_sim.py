from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from multiprocessing import Pool
import random


def dist_from_hyplane(x, w, b):
    '''
    Get the distance of a point x from they hyperplane defined by its intersect and normal
    :param x: point
    :param w: normal vector
    :param b: intersect
    :return: distance
    '''
    return (np.dot(w, x) + b) / np.linalg.norm(w)


def generate_labeled_points(n, dim, neg_gamma=0):
    '''
    Generate a list of linearly separable points with a minimum margin of separation
    :param n: number of points
    :param dim: dimensionality of points
    :param gamma: margin of separation
    :return: points - np.array of linearly separable points; margin - actual margin of separation
    '''
    # Define a random hyperplane in space
    norm = np.random.uniform(low=-1, high=1, size=dim)
    intercept = 0  # np.random.uniform(low=-1, high=1)

    points = []

    i = 0
    while i < n:
        # Get point, label and its distance from hyperplane
        point = np.random.uniform(low=-1, high=1, size=dim)
        dist = dist_from_hyplane(point, norm, intercept)
        if abs(dist) >= gamma and np.sign(dist) == 1:
            label = 1
        elif abs(dist) >= gamma and np.sign(dist) == -1:
            label = -1
        else:
            label = random.choice([-1,1])

        points.append([point, label])
        i += 1

    # Get minimum distance of any points from the plane. this is the margin

    return points


def simulation(n, runs, constant=1, margin=0, train_ratio=0.8, d=2):
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

    all_data = []
    for i in range(runs):
        # Get test data and its gamma, split 80-20 test train
        data = generate_labeled_points(n, neg_gamma=margin, dim=d)

        train_dat, test_dat = train_test_split(data, train_size=train_ratio, test_size=(1-train_ratio))
        # Separate train points from labels
        train_points = [x[0] for x in train_dat]
        train_labels = [x[1] for x in train_dat]

        # Separate test points from their labels
        test_points = [x[0] for x in test_dat]
        test_labels = [x[1] for x in test_dat]

        # Train and test with single SVM
        svm = SVC(kernel="linear", C=constant)
        svm.fit(train_points, train_labels)
        svm_error = svm.score(test_points, test_labels)

        all_data.append([n, margin, constant, svm_error])

    df = pd.DataFrame(all_data, columns=['n', 'margin', 'constant', 'svm_error'])
    return df


def print_progress_bar (iteration, total, prefix='', suffix='', decimals=2, length=50, fill='█'):
    '''
    Auxillary function. Gives us a progress bar which tracks the completion status of our task. Put in loop.
    :param iteration: current iteration
    :param total: total number of iterations
    :param prefix: string
    :param suffix: string
    :param decimals: float point precision of % done number
    :param length: length of bar
    :param fill: fill of bar
    :return:
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    #############################################################
    # SET PARAMETERS HERE, THEN RUN
    #############################################################
    # Fix n to 100 points
    n = 100
    runs = 100

    c_lo = 1
    c_hi = 100
    c_step = 10

    gamma_lo = 0
    gamma_hi = 1
    gamma_step = 0.1

    # File name to write
    file = "dat2.csv"

    ############################################################
    tasks = []
    total = 0
    for c in np.arange(c_lo, c_hi, c_step):
        for gamma in np.arange(gamma_lo, gamma_hi, gamma_step):
            tasks.append((n, runs, c, gamma, ))
            total += 1

    # Progress bar stuff
    iteration = 0
    prefix = "Simulation"
    suffix = "Complete"
    print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    # Send our tasks to the process pool, as they complete append their results to data
    data = []
    with Pool(processes=3) as pool:
        results = [pool.apply_async(simulation, args=t) for t in tasks]
        for r in results:
            iteration += 1
            data.append(r.get())
            print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    print("Writing data...")
    df = pd.concat(data)
    df.to_csv(file, sep=',', index=False)


