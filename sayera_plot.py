import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "svc_rbf_lin_c_gamma_500runs.csv"
zvar = 'test_error'
df = pd.read_csv("data/{}".format(file))

max_df = df.loc[df['machine_param'] == 1.9]
mean_df = max_df.groupby(['point_param'], as_index=False)['test_error'].mean()
sd_df = max_df.groupby(['point_param'], as_index=False)['test_error'].std()


xx = mean_df.as_matrix(['point_param'])
yy = mean_df.as_matrix(['test_error'])
sd = sd_df.as_matrix(['test_error'])


fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(xx, yy)

for i in np.arange(0, len(sd)):
    ax.scatter([xx[i], xx[i]], [yy[i] + sd[i], yy[i] - sd[i]],
               marker="_", c='y')
plt.title("SVC LINEAR KERNEL LINEAR POINTS")

ax.set_xlabel("Gamma Value")
ax.set_ylabel("Test Error")
plt.show()
'''
########################################################################3

file = "nusvm_lin_lin_nu_gamma_500runs.csv"
df = pd.read_csv("data/{}".format(file))
df = df.loc[df['test_error'] >= 0]
fig = plt.figure(1)
idx = 1
for i in np.arange(0.04, 1, 0.2):
    max_df = df.loc[df['machine_param'] == i]
    mean_df = max_df.groupby(['point_param'], as_index=False)['test_error'].mean()
    sd_df = max_df.groupby(['point_param'], as_index=False)['test_error'].std()

    xx = mean_df.as_matrix(['point_param'])
    print(xx)
    yy = mean_df.as_matrix(['test_error'])
    sd = sd_df.as_matrix(['test_error'])

    ax = fig.add_subplot(1, 5, idx)
    idx += 1
    ax.scatter(xx, yy)

    for i in np.arange(0, len(sd)):
        ax.scatter([xx[i], xx[i]], [yy[i] + sd[i], yy[i] - sd[i]],
                   marker="_", c='y')
plt.show()



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

'''