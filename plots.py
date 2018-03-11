import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":
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

