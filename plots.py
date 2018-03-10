import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("dat.csv")
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

