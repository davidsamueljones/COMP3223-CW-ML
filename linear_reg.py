import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as spy
import scipy.linalg
import seaborn as sns


def get_test_y(xs, e):
    return np.sin(xs) + np.random.normal(0, e, xs.shape[0])

# 100000000


def lin_reg(xs, ys, p, ws_init=None, lr=0.000001, l2=-0.05, num_itr=10000000):
    if not ws_init:
        ws_init = np.random.uniform(-0.1, 0.1, p+1)
    dm = get_design_matrix(xs, p)
    ws = grad_descent(dm, ys, ws_init, lr, l2, num_itr)
    print(ws)
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


####################################################
# MAIN
####################################################
if __name__ == "__main__":
    np.random.seed(0)

    xs = np.arange(0, 2 * np.pi, 0.1)
    ys = get_test_y(xs, 0.25)
    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    data_ax.scatter(xs, ys, color='blue')


    # ! Section 4
    # --- Question 1
    
    #ys_new, ws = lin_reg(xs, ys, 3)
    #print(ws)
    #data_ax.plot(xs, ys_new, color='red')

    # Test line with taylor expansion
    w_taylor = [0, 1, 0, -1.0/6.0, 0, 1/120.0, 0, -1/5040.0, 0, 1/362880.0, 0, -1/39916800, 0, 1/6227020800.0]
    ys_taylor = get_design_matrix(xs, 13).dot(w_taylor)
    data_ax.plot(xs, ys_taylor, color='black', linestyle='--')
    



    # --- Question 2
    p = 11
    l2 = 0
    dm = get_design_matrix(xs, p)
    w_analytical = np.linalg.inv(dm.T @ dm + l2 * np.identity(p+1)) @ dm.T @ ys
    ys_analytical = dm.dot(w_analytical)
    data_ax.plot(xs, ys_analytical, color='green')
    print(w_analytical)
    # Make sure we see any issues before showing figures
    sys.stdout.flush()
    sys.stderr.flush()
    # Show generated figures
    plt.show()
