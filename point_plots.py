import matplotlib.pyplot as plt
from point_generator import *

def plot_distro(type, binary, noise):

    l_points = []
    if type == 'linear':
        l_points = linear_labeled_points(n=1000, dim=2, gamma=noise)
    elif type == 'islands':
        l_points = island_labeled_points(n=1000, dim=2, gamma=noise)
    elif type == 'gauss':
        l_points = gaussian_labeled_points(n=1000, n_means=4, dim=2, gamma=noise, binary=binary)
    elif type == 'random_walk':
        l_points = random_walks(n=1000, p=noise, grids=20)
    else:
        l_points = None

    points = [x[0] for x in l_points]
    labels = [x[1] for x in l_points]

    colors = []
    if binary:
        for x in labels:
            if x == 1:
                colors.append("r")
            else:
                colors.append("b")
    else:
        cls = "rbgcmykw"
        for x in labels:
            colors.append(cls[x])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, x in enumerate(points):
        ax.scatter(x[0], x[1], c=colors[i])

    plt.show()