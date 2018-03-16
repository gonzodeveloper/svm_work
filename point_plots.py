from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from svm_sim import generate_gaussian_labeled_points, generate_labeled_points, generate_island_labeled_points

if __name__ == "__main__":
    type = sys.argv[1]
    margin = float(sys.argv[3])
    binary = True if sys.argv[2] == "binary" else False
    if type == 'linear':
        l_points = generate_labeled_points(n=1000, dim=2, gamma=margin)
    elif type == 'islands':
        l_points = generate_island_labeled_points(n=1000, dim=2, gamma=margin)
    elif type == 'gauss':
        l_points = generate_gaussian_labeled_points(n=1000, n_means=4, dim=2, gamma=margin, binary=binary)
    else:
        exit(1)
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