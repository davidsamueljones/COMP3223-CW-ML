import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as spy
import scipy.linalg
import sklearn.model_selection
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def get_test_y(xs, e):
    return np.sin(xs) + np.random.normal(0, e, xs.shape[0])


def lin_reg(xs, ys, p, ws_init=None, lr=0.0001, l2=0, num_itr=10000000):
    if not ws_init:
        ws_init = np.random.uniform(-0.1, 0.1, p + 1)
    dm = get_design_matrix(xs, p)
    ws = grad_descent(dm, ys, ws_init, lr, l2, num_itr)
    return dm.dot(ws), ws


def get_design_matrix(xs, p):
    size = xs.shape[0]
    xs = xs.reshape(size, 1)
    dm = np.ones((size, 1))
    dm = np.hstack((dm, xs))
    for exp in range(2, p + 1, 1):
        dm = np.hstack((dm, xs**exp))
    return dm


def calc_loss(dm, ys, ws, l2):
    rss = (dm.T.dot(ys - dm.dot(ws)) ** 2).mean()
    C_w = l2 * np.sqrt(ws.dot(ws))
    return rss + C_w


def grad_loss(dm, ys, ws, l2):
    N, p = dm.shape
    grad_rss = (-2 / N) * dm.T.dot(ys - dm.dot(ws))
    grad_C_w = l2 * 2 * ws
    return grad_rss + grad_C_w


def grad_descent(dm, ys, ws_init, lr, l2, num_itr):
    N, p = dm.shape
    ws_history = []
    loss_history = []
    ws = ws_init

    for itr in range(num_itr):
        ws_history.append(ws)
        loss = calc_loss(dm, ys, ws, l2)
        loss_history.append(loss)
        ws = ws - lr * grad_loss(dm, ys, ws, l2)

    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    data_ax.plot(range(num_itr), loss_history)

    return ws.reshape(p, 1)


def get_analytic_w(dm, ys, l2):
    p = dm.shape[1] - 1
    return np.linalg.inv(dm.T @ dm + l2 * np.identity(p + 1)) @ dm.T @ ys


def calc_rss(dm, ys, w):
    return np.square(ys - dm.dot(w)).mean()


def rss_plots():
        # Create a plot where l2 = 0
    l20_fig, l20_ax = plt.subplots(ncols=1, nrows=1)
    l20_ax.scatter(xs_test, ys_test, color='black', alpha=0.25)
    l20_ax.plot(xs, ys_taylor, color='black', linestyle='--')
    line_idx = 0
    l2 = 0
    li = (np.abs(l2s - l2)).argmin()
    for pi, p in enumerate(ps):
        dm = get_design_matrix(xs, p)
        dm_train = get_design_matrix(xs_train, p)
        w_analytic = get_analytic_w(dm_train, ys_train, l2)
        if p in [1, 2, 3, 5, 11, 19]:
            l20_ax.plot(xs, dm.dot(w_analytic), color=colors[line_idx],
                        label='P={:2d} l2={:+0.2f} rss={:0.4f}'.format(p, l2, Z[pi][li]))
            line_idx += 1
    l20_ax.legend()

    # Create a plot of the optimal l2s
    data_fig, l2_star_ax = plt.subplots(ncols=1, nrows=1)
    l2_star_ax.scatter(xs_test, ys_test, color='black', alpha=0.25)
    l2_star_ax.plot(xs, ys_taylor, color='black', linestyle='--', label='')
    line_idx = 0
    for pi, p in enumerate(ps):
        dm = get_design_matrix(xs, p)
        dm_train = get_design_matrix(xs_train, p)
        li = np.argmin(Z[pi])
        l2 = l2s[li]
        w_analytic = get_analytic_w(dm_train, ys_train, l2)
        if p in [1, 2, 3, 5, 11, 19]:
            l2_star_ax.plot(xs, dm.dot(w_analytic), color=colors[line_idx],
                            label='P={:2d} l2={:+0.2f} rss={:0.4f}'.format(p, l2, Z[pi][li]))
            line_idx += 1
    l2_star_ax.legend()
    l20_ax.set_xlim(l2_star_ax.get_xlim())
    l20_ax.set_ylim(l2_star_ax.get_ylim())

    # Create a plot of some l2s for a specific p
    single_p_fig, single_p_ax = plt.subplots(ncols=1, nrows=1)
    single_p_ax.scatter(xs_test, ys_test, color='black', alpha=0.25)
    single_p_ax.plot(xs, ys_taylor, color='black', linestyle='--')
    p = 15
    pi = (np.abs(ps - p)).argmin()
    for line_idx, l2 in enumerate(np.arange(-1, 1 + 1 / 3, 1 / 3)):
        li = (np.abs(l2s - l2)).argmin()
        dm = get_design_matrix(xs, p)
        dm_train = get_design_matrix(xs_train, p)
        w_analytic = get_analytic_w(dm_train, ys_train, l2)
        single_p_ax.plot(xs, dm.dot(w_analytic), color=colors[line_idx],
                         label='P={:2d} l2={:+0.2f} rss={:0.4f}'.format(p, l2, Z[pi][li]))
    single_p_ax.legend()
    single_p_ax.set_xlim(l2_star_ax.get_xlim())
    single_p_ax.set_ylim(l2_star_ax.get_ylim())


def rss_plot_3d():
    rss_fig = plt.figure()
    rss_ax = rss_fig.add_subplot(111, projection='3d')
    rss_ax.set_xlabel('l2')
    rss_ax.set_ylabel('p')
    rss_ax.set_zlabel('ln(rss)')
    # Set y axis to only be integer values
    rss_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Plot data
    rss_ax.plot_surface(X, Y, np.log(Z), rstride=1, cstride=1, cmap=cm.plasma)


####################################################
# MAIN
####################################################
if __name__ == "__main__":
    np.random.seed(0)

    colors = ['red', 'green', 'blue', 'purple',
              'pink', 'orange', 'cyan', 'gray', 'yellow']

    xs = np.arange(0, 2 * np.pi, 0.01)
    ys = get_test_y(xs, 0.25)
    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    data_ax.scatter(xs, ys, color='black', alpha=0.25)

    # Test line with taylor expansion
    w_taylor = [0, 1, 0, -1.0 / math.factorial(3), 0, 1 / math.factorial(5), 0, -1 / math.factorial(7),
                0, 1 / math.factorial(9), 0, -1 /
                math.factorial(11), 0, 1 / math.factorial(13),
                0, -1 / math.factorial(15), 0, 1 / math.factorial(17), 0, -1 / math.factorial(19)]
    ys_taylor = get_design_matrix(xs, 19).dot(w_taylor)
    data_ax.plot(xs, ys_taylor, color='black', linestyle='--')

    # ! Section 4.1
    # --- Question 1
    p = 3
    l2 = 0
    # Known gradient descent ways
    for i, itr in enumerate([10000, 100000, 1000000, 10000000]):
        # ys_new, ws = lin_reg(xs, ys, p, l2=l2, num_itr=itr)
        # dm = get_design_matrix(xs, p)
        # ys_new = dm.dot(wss[i])
        # data_ax.plot(
        #    xs, ys_new, color=colors[i], label='num_itr = {}'.format(itr))
        # print(ws)
        # These take a while, so flush in case of failures
        sys.stdout.flush()
    data_ax.legend()

    # --- Question 2
    dm = get_design_matrix(xs, p)
    w_analytic = get_analytic_w(dm, ys, l2)
    ys_analytic = dm.dot(w_analytic)
    # data_ax.plot(xs, ys_analytic, color='green')
    # print(w_analytic)

    # --- Question 3
    # Split into training and test data
    xs_train, xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(
        xs, ys, test_size=0.25, shuffle=True)
    train_order = xs_train.argsort()
    test_order = xs_test.argsort()
    # Set the test range across many p's and many
    ps = np.arange(1, 20, 1)
    l2_step = 0.01
    l2s = np.arange(-1, 1 + l2_step, l2_step)
    X, Y = np.meshgrid(l2s, ps)
    Z = np.zeros((len(ps), len(l2s)))
    results = np.empty((3, len(ps) * len(l2s)))
    for pi, p in enumerate(ps):
        dm_train = get_design_matrix(xs_train, p)
        dm_test = get_design_matrix(xs_test, p)
        for li, l2 in enumerate(l2s):
            w_analytic = get_analytic_w(dm_train, ys_train, l2)
            # Test result
            rss = calc_rss(dm_test, ys_test, w_analytic)
            idx = pi * len(l2s) + li
            results[0, idx], results[1, idx], results[2, idx] = p, l2, rss
            Z[pi][li] = rss

    # Make some helpful plots
    rss_plots()
    rss_plot_3d()

    # ! Section 4.2
    # Question 1
    ps = range(0, 6)
    l2 = 0
    groups = 10
    samples = 15
    xs_train_i = np.empty((groups, samples))
    ys_train_i = np.empty((groups, samples))
    ys_hat = np.empty((len(ps), groups, len(xs)))
    ys_hat_test = np.empty((len(ps), groups, len(xs_test)))
    train_rss = np.empty((len(ps), groups))
    test_rss = np.empty((len(ps), groups))

    for gi in range(groups):
        # Create smaller training group, use existing test set
        sampled = np.random.choice(len(xs_train), samples)
        xs_train_i[gi] = xs_train[sampled]
        ys_train_i[gi] = ys_train[sampled]
        # Test for each test degree
        for pi, p in enumerate(ps):
            # Find w_hat from training data
            dm_train = get_design_matrix(xs_train_i[gi], p)
            w_hat = get_analytic_w(dm_train, ys_train_i[gi], l2)

            # Verify training error
            train_rss[pi][gi] = calc_rss(dm_train, ys_train_i[gi], w_hat)

            # Test using test group (calculate data and MSE)
            dm_test = get_design_matrix(xs_test, p)
            ys_hat_test[pi][gi] = dm_test.dot(w_hat)
            test_rss[pi][gi] = calc_rss(dm_test, ys_test, w_hat)

            # Get data for plotting mean lines
            dm = get_design_matrix(xs, p)
            ys_hat[pi][gi] = dm.dot(w_hat)

    # Calculations on test results
    test_errs = np.empty(len(ps))
    train_errs = np.empty(len(ps))
    bias_sqs = np.empty(len(ps))
    vars = np.empty(len(ps))
    avg_fig, avg_ax = plt.subplots(ncols=1, nrows=1)
    for pi, p in enumerate(ps):
        test_errs[pi] = np.mean(test_rss[pi])
        train_errs[pi] = np.mean(train_rss[pi])
        bias_sqs[pi] = np.mean((np.mean(ys_hat_test[pi]) - ys_test) ** 2)
        vars[pi] = np.mean(np.var(ys_hat_test[pi], 1))
        avg_ax.plot(xs, np.mean(ys_hat[pi], 0))

    bias_fig, bias_ax = plt.subplots(ncols=1, nrows=1)
    bias_ax.plot(ps, bias_sqs, color='r', label='Bias^2')
    bias_ax.plot(ps, vars, color='b', label='Variance')
    bias_ax.plot(ps, bias_sqs + vars, color='g', label='Bias^2 + Variance')
    bias_ax.set_xlabel('Complexity (p)')
    bias_ax.legend()
    bias_ax.set_xlim(0, 5);

    error_fig, error_ax = plt.subplots(ncols=1, nrows=1)
    error_ax.plot(ps, test_errs, color='blue', label='Train RSS')
    error_ax.plot(ps, train_errs, color='red', label='Test RSS')
    error_ax.set_xlabel('Complexity (p)')
    error_ax.set_ylabel('RSS')
    error_ax.legend()
    error_ax.set_xlim(0, 5);

    # Make sure we see any issues before showing figures
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Show generated figures
    plt.show()

    # wss3 = np.array([[-0.00693982, -0.07747356, -0.05611875, 0.00915971],
    #                      [-0.02287977, 0.07479174, -0.01647654, -0.00350449],
    #                      [0.36323588, 0.18265207, -0.15737371, 0.01610813],
    #                      [0.50416634, 0.88155106, -0.52559247, 0.05869679],
    #                      [-0.02996662, 1.67210794, -0.80225586, 0.08579929]])
    #                  # lr = 0.000001
    # wss5 = np.array([[-0.01395734, -0.08533845, -0.05350415, 0.0986727, -0.03606846, 0.00353482],
    #                  [0.06825513, -0.05645454, 0.03819409, -0.06325869, 0.01552034, -0.0010362],
    #                  [0.10295992, 0.04689671, 0.1474181, -0.02578695, -0.0167937, 0.00274448],
    #                  [0.25118434, 0.18737344, 0.06991354, -0.03370698, -0.01047395, 0.00214352],
    #                  [0.38744715, 0.43345664, 0.23409419, -0.273299, 0.05499004, -0.00308328]])
    #                  # lr = 0.0000001
    # wss7 = np.array([[-0.01492817, -0.08732454, -0.05849464, 0.08589677, -0.05910787, 0.06459234, 0.04692919, -0.00965749],
    #                  [0.02111003, -0.07700329, 0.04501331, 0.02519763,
    #                      0.05353557, -0.03443287, 0.00966099, -0.00094037],
    #                  [-0.04141118, 0.04303199, -0.01632999, -0.06530101, - 0.07838251, 0.06302011, -0.01372748, 0.00094995],
    #                  [0.0464111, -0.01884277, -0.02755436, -0.00249186,
    #                      0.06767553, 0.046546, -0.02447793, 0.00247652],
    #                  [0.08492205, 0.04462042, -0.05801445, 0.01300276, -0.03841797, 0.00761882, 0.00079397, -0.00017122]])
    #                  # lr = 0.00000000001
    #                  
    #                  
