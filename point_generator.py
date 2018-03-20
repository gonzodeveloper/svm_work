import numpy as np
import random
from scipy.spatial import distance
import math


def dist_from_hyplane(x, w, b):
    '''
    Get the distance of a point x from they hyperplane defined by its intersect and normal
    :param x: point
    :param w: normal vector
    :param b: intersect
    :return: distance
    '''
    return (np.dot(w, x) + b) / np.linalg.norm(w)


def linear_labeled_points(n, dim, gamma=0):
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


def island_labeled_points(n, dim, gamma, n_islands=2, radius=.5):
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
    return points

def gaussian_labeled_points(n, dim, gamma, n_means=4, binary=True):
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

def random_walks(n, p, grids=20):
    """ Generates 'n' number of points and labels by random walk distribution in 2d-space.
        'p' is a parameter ranging in [0,inf). 0 minimum noise inf maximum noise.
        'grids' is the number of walks occuring along each dimension.
    """
    walks = np.zeros((grids,grids))
    cox = random.randint(0,grids-1)
    coy = random.randint(0,grids-1)
    for x in range(0,cox):
        walks[cox-1-x,coy] = walks[cox-x,coy]+1-2*random.randint(0,1)
    for x in range(cox+1,grids):
        walks[x,coy] = walks[x-1,coy]+1-2*random.randint(0,1)
    for y in range(0,coy):
        walks[cox,coy-1-y] = walks[cox,coy-y]+1-2*random.randint(0,1)
        for x in range(0,cox):
            if walks[cox-x,coy-1-y] == walks[cox-1-x,coy-y]:
                walks[cox-1-x,coy-1-y] = walks[cox-x,coy-1-y]+1-2*random.randint(0,1)
            else:
                walks[cox-1-x,coy-1-y] = (walks[cox-x,coy-1-y]+walks[cox-1-x,coy-y])/2
        for x in range(cox+1,grids):
            if walks[x-1,coy-1-y] == walks[x,coy-y]:
                walks[x,coy-1-y] = walks[x-1,coy-1-y]+1-2*random.randint(0,1)
            else:
                walks[x,coy-1-y] = (walks[x-1,coy-1-y]+walks[x,coy-y])/2
    for y in range(coy+1,grids):
        walks[cox,y] = walks[cox,y-1]+1-2*random.randint(0,1)
        for x in range(0,cox):
            if walks[cox-x,y] == walks[cox-1-x,y-1]:
                walks[cox-1-x,y] = walks[cox-x,y]+1-2*random.randint(0,1)
            else:
                walks[cox-1-x,y] = (walks[cox-x,y]+walks[cox-1-x,y-1])/2
        for x in range(cox+1,grids):
            if walks[x-1,y] == walks[x,y-1]:
                walks[x,y] = walks[x-1,y]+1-2*random.randint(0,1)
            else:
                walks[x,y] = (walks[x-1,y]+walks[x,y-1])/2
    maximum = 0
    minimum = 0
    for i in range(grids):
        for j in range(grids):
            if walks[i,j] > maximum:
                maximum = walks[i,j]
            if walks[i,j] < minimum:
                minimum = walks[i,j]
    for i in range(grids):
        for j in range(grids):
            walks[i,j] -= (maximum+minimum)/2.0
            walks[i,j] /= (maximum-minimum)/2.0
            walks[i,j] = np.sign(walks[i,j])*math.pow(abs(walks[i,j]),p)
    points = []
    for i in range(n):
        x = np.random.uniform(low=-1, high=1, size=2)
        a = x[0]+1
        b = x[1]+1
        a *= grids/2.0
        b *= grids/2.0
        a = int(math.floor(a))
        b = int(math.floor(b))
        chance = [0.5+0.5*walks[a,b], 0.5-0.5*walks[a,b]]
        label = np.random.choice([1, -1], p=chance)
        points.append([x,label])
    return points
