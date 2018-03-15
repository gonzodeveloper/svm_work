from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from svm_sim import generate_gaussian_labeled_points, generate_labeled_points

if __name__ == "__main__":
    type = sys.argv[1]
    margin = float(sys.argv[2])
    if type == 'linear':
        l_points = generate_labeled_points(n=1000, dim=2, gamma=margin)
    elif type == 'gauss':
        l_points = generate_gaussian_labeled_points(n=1000, dim=2, gamma=margin)
    else:
        exit(1)
    points = [x[0] for x in l_points]
    labels = [x[1] for x in l_points]

    colors = []
    for x in labels:
        if x == 1:
            colors.append("r")
        else:
            colors.append("b")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, x in enumerate(points):
        ax.scatter(x[0], x[1], c=colors[i])

    plt.show()