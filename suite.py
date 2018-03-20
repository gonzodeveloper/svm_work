from simulator import simulation
from error_plots import plot_error
from point_plots import plot_distro
import numpy as np
import pandas as pd
from multiprocessing import Pool
import re
import readline


def print_progress_bar (iteration, total, prefix='', suffix='', decimals=2, length=50, fill='█'):
    '''
    Auxillary function. Gives us a progress bar which tracks the completion status of our task. Put in loop.
    :param iteration: current iteration
    :param total: total number of iterations
    :param prefix: string
    :param suffix: string
    :param decimals: float point precision of % done number
    :param length: length of bar
    :param fill: fill of bar
    :return:
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def run_sim(model, n, dim, runs, noise, param, ratio, dist, data_class, kern, file):
    noise_min, noise_max, noise_step = noise
    param_min, param_max, param_step = param
    ratio_min, ratio_max, ratio_step = ratio

    tasks = []
    total = 0
    for r in np.arange(ratio_min, ratio_max, ratio_step):
        for p in np.arange(param_min, param_max, param_step):
            for g in np.arange(noise_min, noise_max, noise_step):
                tasks.append((n, dim, dist, data_class, g, model, p, kern, runs, r, ))
                total += 1

    # Progress bar stuff
    iteration = 0
    prefix = "Simulation"
    suffix = "Complete"
    print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    # Send our tasks to the process pool, as they complete append their results to data
    data = []
    with Pool(processes=3) as pool:
        results = [pool.apply_async(simulation, args=t) for t in tasks]
        for r in results:
            iteration += 1
            data.append(r.get())
            print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    print("Writing data...")
    df = pd.concat(data)
    df.to_csv("data/{}".format(file), sep=',', index=False)


def simulation_options(error):
    print("\rPick a model \n"
          "1. SVC \n"
          "2. NuSVC \n"
          "3. Logistic Regression")
    ins = input(">>>> ")
    if ins == "1":
        model = "svc"
    elif ins == "2":
        model = "nusvc"
    elif ins == "3":
        model = "lr"
    else:
        model = None
        print(error)
        exit(1)
    print("\nPick a Kernel or Logistic Regression Scheme\n"
          "1. Linear kernel\n"
          "2. Polynomial kernel\n"
          "3. RBF kernel\n"
          "4. OvR scheme\n"
          "5. Multiclass scheme")
    ins = input(">>>> ")
    if ins == "1":
        kern = 'linear'
    elif ins == "2":
        kern = 'poly'
    elif ins == "3":
        kern = 'rbf'
    elif ins == "4":
        kern = 'ovr'
    elif ins == "5":
        kern = 'multinomial'
    else:
        kern = None
        print(error)
        exit(1)
    print("\nPick a data distribution\n"
          "1. Linear\n"
          "2. \"Islands\"\n"
          "3. Gaussian n-means\n"
          "4. Random Walk")
    ins = input(">>>> ")
    if ins == "1":
        dist = 'linear'
    elif ins == "2":
        dist = 'islands'
    elif ins == "3":
        dist = 'gauss'
    elif ins == "4":
        dist = 'random_walk'
    else:
        dist = None
        print(error)
        exit(1)
    print("\nPick a label distribution (multi-class only availible for gaussian n-means data)\n"
          "1. Binary\n"
          "2. Multi-class")
    ins = input(">>>> ")
    if ins == "1":
        data_class = 'binary'
    elif ins == "2":
        data_class = 'multi'
    else:
        print(error)
        exit(1)
    print("\nEnter a the number of data points you would like to simulate")
    n = int(input(">>>> "))
    print("\nEnter the number of runs for each step of the simulation (default: 100)")
    runs = int(input(">>>> "))
    print("\nEnter the number of dimensions for data (default: 2)")
    dim = int(input(">>>> "))

    print("\n\nFor the simulation we have three independent variables that we may plot against error:\n"
          "model parameter, noise in data, sparsity of data. For experiment sake, one of these must\n"
          "be fixed. In the following prompts you will be asked to set a range and step size for \n"
          "these three variables. To set the range enter 3 comma+space separated values: min, max and step\n"
          "e.g. 0.1, 1, 0.05. Whichever value you would like to fix just enter a single value\n")

    print("Enter parameter range (c for SVC or Logistic Regression, nu for NuSVC)")
    ins = input(">>>> ")
    vals = [float(x) for x in re.split(", ", ins)]
    if len(vals) == 1:
        param = (vals[0], vals[0] + 1, 100)
    else:
        param = (vals[0], vals[1], vals[2])
    print("\nEnter noise range (gamma value for linear data, distance between gaussian clusters, or \n"
          "p parameter in random walk)")
    ins = input(">>>> ")
    vals = [float(x) for x in re.split(", ", ins)]
    if len(vals) == 1:
        noise = (vals[0], vals[0] + 1, 100)
    else:
        noise = (vals[0], vals[1], vals[2])
    print("\nEnter range for training ratio (represents data sparsity)")
    ins = input(">>>> ")
    vals = [float(x) for x in re.split(", ", ins)]
    if len(vals) == 1:
        train_ratio = (vals[0], vals[0] + 1, 100)
    else:
        train_ratio = (vals[0], vals[1], vals[2])
    print("\nEnter file name to save data")
    file = input(">>>> ")

    print("\nRunning simmulation")
    run_sim(model, n, dim, runs, noise, param, train_ratio, dist, data_class, kern, file)


def error_polt_options(error):
    print("\nEnter the name of the file you would like to load.")
    file = input(">>>> ")
    print("\nEnter a title for your plot")
    title = input(">>>> ")
    print("\nWhich variable would you like for the x-axis?\n"
          "1. Model parameter (c or nu)\n"
          "2. Distribution noise (gamma or p value)\n"
          "3. Train ratio")
    ins = input(">>>> ")
    if ins == '1':
        xvar = 'machine_param'
    elif ins == '2':
        xvar = 'point_param'
    elif ins == '3':
        xvar = 'train_ratio'
    else:
        print(error)
        exit(1)
    print("\nWhich variable would you like for the y-axis?\n"
          "1. Model parameter (c or nu)\n"
          "2. Distribution noise (gamma or p value)\n"
          "3. Train ratio")
    ins = input(">>>> ")
    if ins == '1':
        yvar = 'machine_param'
    elif ins == '2':
        yvar = 'point_param'
    elif ins == '3':
        yvar = 'train_ratio'
    else:
        print(error)
        exit(1)
    print("z-axis represents test error!")
    plot_error(file, title, xvar, yvar)


def sample_plot_options(error):
    print("\nPick a distibution to sample plot\n"
          "1. Linear\n"
          "2. \"Islands\"\n"
          "3. Gaussian n-means\n"
          "4. Random walk")
    ins = input(">>>> ")
    if ins == "1":
        type = 'linear'
    elif ins == "2":
        type = 'islands'
    elif ins == "3":
        type = 'gauss'
    elif ins == "4":
        type = 'random_walk'
    else:
        print(error)
        exit(1)

    print("\nBinary or multi-class data? (multi-class only availible for n-means)\n"
          "1. Binary\n"
          "2. Multiclass")
    ins = input(">>>> ")
    if ins == "1":
        binary = True
    else:
        binary = False
    print("\nEnter a value for noise in data \n"
          "\tgamma for linear/islands \n"
          "\tinter-cluster distance for gauss\n"
          "\t p value for random walk")
    noise = float(input(">>>> "))
    plot_distro(type, binary, noise)


if __name__ == "__main__":
    error = "Invalid Input"
    print("What would you like to do?\n"
          "1. Run Simulation\n"
          "2. Plot Error\n"
          "3. PLot Sample distributions")
    ins = input(">>>> ")
    if ins == "1":
        simulation_options(error)
    elif ins == "2":
        error_polt_options(error)
    elif ins == "3":
        sample_plot_options(error)
    else:
        exit()

