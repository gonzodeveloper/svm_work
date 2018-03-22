import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fail_ratio(series):
    return len(series.loc[series < 0]) / len(series)


def plot_error(file, title, xvar, yvar):
    zvar = 'test_error'
    df = pd.read_csv("data/{}".format(file))
    non_failures = df.loc[df[zvar] >= 0]
    means_df = non_failures.groupby([xvar, yvar], as_index=False)[zvar].mean()
    sd_df = non_failures.groupby([xvar, yvar], as_index=False)[zvar].std()

    xx = means_df.as_matrix([xvar])
    yy = means_df.as_matrix([yvar])
    zz = means_df.as_matrix([zvar])
    z_sd = sd_df.as_matrix([zvar])

    fig = plt.figure(1)
    ax = fig.add_subplot(121, projection='3d')
    plt.title(title)
    ax.scatter(xx, yy, zz)

    for i in np.arange(0, len(z_sd)):
        ax.scatter([xx[i], xx[i]], [yy[i], yy[i]], [zz[i] + z_sd[i], zz[i] - z_sd[i]],
                marker="_", c='y')

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_zlabel(zvar)

    if len(df.loc[df[zvar] < 0]) > 0:
        ax = fig.add_subplot(122, projection='3d')
        print(df)
        fails = df.groupby([xvar, yvar], as_index=False)[zvar].agg(fail_ratio)
        xx = fails.as_matrix([xvar])
        yy = fails.as_matrix([yvar])
        zz = fails.as_matrix([zvar])

        ax.scatter(xx, yy, zz, c='r')
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_zlabel("Fail Rate")
    plt.show()
