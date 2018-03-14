import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from svm_sim import generate_gaussian_labeled_points

if __name__ == "__main__":

    l_points = generate_gaussian_labeled_points(n_points=1000, n_means=3, sd=.1,dim=2)
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
    '''
    ############################################
    # WHICH FILE TO PLOT?
    ############################################
    file = "dat.csv"
    ############################################
    df = pd.read_csv("data/{}".format(file))
    df = df.groupby(['margin', 'constant'], as_index=False)['svm_error'].mean()
    print(df)
    xx = df.as_matrix(['margin'])
    yy = df.as_matrix(['constant'])
    zz = df.as_matrix(['svm_error'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx, yy, zz)
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('error')
    plt.show()

    '''