import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from svm_sim import generate_gaussian_labeled_points, generate_labeled_points

if __name__ == "__main__":

    ############################################
    # WHICH FILE TO PLOT?
    ############################################
    file = "svc_gamma_pos_to_neg.csv"
    ############################################
    df = pd.read_csv("data/{}".format(file))
    df = df.groupby(['margin', 'constant'], as_index=False)['svc_error'].mean()
    xx = df.as_matrix(['margin'])
    yy = df.as_matrix(['constant'])
    zz = df.as_matrix(['svc_error'])

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx, yy, zz)
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('svc error')

    df = df.groupby(['margin', 'constant'], as_index=False)['nu_error'].mean()
    xx = df.as_matrix(['margin'])
    yy = df.as_matrix(['constant'])
    zz = df.as_matrix(['nu_error'])

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx, yy, zz)
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('nu error')

    plt.show()
