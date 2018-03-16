from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from multiprocessing import Pool
import random
from scipy.spatial import distance


def dist_from_hyplane(x, w, b):
    '''
    Get the distance of a point x from they hyperplane defined by its intersect and normal
    :param x: point
    :param w: normal vector
    :param b: intersect
    :return: distance
    '''
    return (np.dot(w, x) + b) / np.linalg.norm(w)


def generate_labeled_points(n, dim, gamma=0):
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
        if gamma < 0:
            neg_gamma = abs(gamma)
            if abs(dist) >= neg_gamma and np.sign(dist) == 1:
                label = 1
            elif abs(dist) >= neg_gamma and np.sign(dist) == -1:
                label = -1
            else:
                label = random.choice([-1,1])
        else:
            if abs(dist) >= gamma and np.sign(dist) == 1:
                label = 1
            elif abs(dist) >= gamma and np.sign(dist) == -1:
                label = -1
            else:
                continue

        points.append([point, label])
        i += 1

    # Get minimum distance of any points from the plane. this is the margin

    return points


def generate_island_labeled_points(n, dim, gamma, n_islands=2, radius=.5):
    """ Generate n_points number of dim-dimensional points and n_means number of dim-dimensional means
        Label the points based on the minimum distance from the means

     """
    means = np.random.uniform(low=-1, high=1, size=(n_islands, dim))
    points = []
    i = 0
    avg = []
    while i < n:
        point = np.random.uniform(low=-1, high=1, size=dim)
        # Find the distance of each point from the means
        distances = [distance.euclidean(mean, point) for mean in means]

        min_dist = min(distances)
        avg.append(min_dist)
        # Negative gamma
        if gamma < 0:
            if min_dist > (radius + abs(gamma)):
                label = 1
            elif min_dist < radius:
                label = -1
            else:
                label = random.choice([-1, 1])
        else:
            if min_dist > (radius + gamma):
                label = 1
            elif min_dist < radius:
                label = -1
            else:
                continue
        points.append([point, label])
        i += 1
    print(np.average(avg))
    return points

def generate_gaussian_labeled_points(n, n_means, dim, gamma, binary=True):
    groups = []
    idx = 0
    # Generate a number of groups on the plane that are separated by at least the "margin's"
    while idx < n_means:
        mean = np.random.uniform(low=-1, high=1, size=dim)
        obeys_margin = True
        for x in groups:
            if distance.euclidean(mean, x['mean']) < gamma:
                obeys_margin = False
        if obeys_margin:
            std_dev = np.random.uniform(low=.1, high=.3)
            label = random.choice([-1, 1]) if binary else idx
            group = {'mean': mean, 'sd': std_dev, 'label': label }
            groups.append(group)
            idx += 1
    # Generate points randomly assigned to groups,
    points = []
    for i in range(n):
        # Pick a random group
        group = random.choice(groups)
        point = np.random.normal(loc=group['mean'], scale=group['sd'])
        label = group['label']
        points.append([point, label])
    return points

def simulation(n, runs, constant=1, margin=0, train_ratio=0.8, d=2, kern="linear"):
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
        data = generate_labeled_points(n, gamma=margin, dim=d)

        train_dat, test_dat = train_test_split(data, train_size=train_ratio, test_size=(1-train_ratio))
        # Separate train points from labels
        train_points = [x[0] for x in train_dat]
        train_labels = [x[1] for x in train_dat]

        # Separate test points from their labels
        test_points = [x[0] for x in test_dat]
        test_labels = [x[1] for x in test_dat]

        # Train and test with single SVM
        svc = SVC(kernel=kern, C=constant)
        svc.fit(train_points, train_labels)
        svc_error = svc.score(test_points, test_labels)

        # Train and test with NuSvc
        try:
            nusvc = NuSVC(kernel=kern, nu=constant)
            nusvc.fit(train_points, train_labels)
            nu_error = nusvc.score(test_points, test_labels)
        except ValueError:
            print(constant)
        all_data.append([n, margin, constant, svc_error, nu_error])

    df = pd.DataFrame(all_data, columns=['n', 'margin', 'constant', 'svc_error', 'nu_error'])
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
    n = 1000
    runs = 10
    train_ratio = 0.2

    constant_lo = 0.1
    constant_hi = 0.9
    constant_step = .02

    gamma_lo = -0.5
    gamma_hi = 0.5
    gamma_step = 0.02

    # File name to write
    file = "svc_gamma_pos_to_neg.csv"

    ############################################################
    tasks = []
    total = 0
    for const in np.arange(constant_lo, constant_hi, constant_step):
        for gamma in np.arange(gamma_lo, gamma_hi, gamma_step):
            c = round(const, 2)
            tasks.append((n, runs, c, gamma, train_ratio, ))
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
    df.to_csv("data/{}".format(file), sep=',', index=False)


