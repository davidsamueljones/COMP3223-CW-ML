import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import scipy.stats as sps
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def get_normal_data(mean, covar, points):
    """ 
    Generate data from a normal Gaussian distribution with a mean vector, covariance matrix and number of data points required.
    Returned is a numpy array of shape [points, 2] where each row is an x, y coordinate and frozen multivariate normal instance.
    """
    mvn = sps.multivariate_normal(mean=mean, cov=covar)
    data = mvn.rvs(size=points)
    sys.stderr.flush()
    # Check if positive-definite manually for debug
    for eig in np.linalg.eigvals(covar):
        if eig < 0:
            print("Err: " + str(eig))
    return np.array(data), mvn


def comp_hist(ax, xa, xb, w, **kwargs):
    ya = calc_y(xa, w)
    yb = calc_y(xb, w)
    # Plot histogram
    ax.hist(ya, alpha=0.5, color='b', density=True)
    ax.hist(yb, alpha=0.5, color='r', density=True)
    if 'title' not in kwargs.keys():
        kwargs['title'] = "W: " + np.array2string(w)
    ax.set_title(kwargs['title'])
    return ya, yb


def calc_y(x, w):
    return w.dot(x)


def calc_R_theta(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def calc_fisher_ratio(ya, yb):
    """
    Calculate the Fisher ratio. TODO

    Parameters:
        ya : Result of class a projection
        yb : Result of class b projection

    Returns:
        Calculated Fisher ratio
    """
    count_a = np.size(ya)
    count_b = np.size(yb)

    top = (np.mean(ya) - np.mean(yb)) ** 2
    bot_lhs = count_a / (count_a + count_b) * np.var(ya)
    bot_rhs = count_b / (count_a + count_b) * np.var(yb)
    ratio = top / (bot_lhs + bot_rhs)
    return ratio

def calc_log_odds(xs, ys, mean_a, mean_b, covar_a, covar_b):
    Lam_a = np.linalg.inv(covar_a)
    Lam_b = np.linalg.inv(covar_b)
    lhs = np.log(np.linalg.det(Lam_a) / np.linalg.det(Lam_b))
    Z = np.empty((X.shape[0], Y.shape[1]))
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            p = np.array([x, y])
            g_a = (p - mean_a).T.dot(Lam_a).dot(p-mean_a) / 2
            g_b = (p - mean_b).T.dot(Lam_b).dot(p-mean_b) / 2
            Z[yi][xi] = lhs + g_b - g_a
    q2a_ax.contour(X, Y, Z, 0, colors='green')
    return Z

####################################################
# MAIN
####################################################
if __name__ == "__main__":
    np.random.seed(0)
    x_range = np.linspace(-10, 20, 100)
    y_range = np.linspace(-10, 20, 100)

    # Dataset A
    count_a = 1000
    mean_a = np.array([5, 5])
    covar_a = np.array([[3, 1], [1, 1]])
    xa, mvn_a = get_normal_data(mean_a, covar_a, count_a)
    xa = xa.T

    # Dataset B
    count_b = count_a
    mean_b = np.array([1, 1])
    covar_b = np.array([[2, 1], [1, 2]])
    xb, mvn_b = get_normal_data(mean_b, covar_b, count_b)
    xb = xb.T

    # Combined Dataset
    comb_x = np.hstack((xa, xb))

    # --- Data plot
    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    data_ax.scatter(xa[0], xa[1], color='b', alpha=0.5)
    data_ax.scatter(xb[0], xb[1], color='r', alpha=0.5)

    # ! Question 1
    # --- (A)
    q1a_fig, q1a_ax = plt.subplots(ncols=2, nrows=2)
    comp_hist(q1a_ax[0][0], xa, xb, np.array([1, 1]))
    comp_hist(q1a_ax[0][1], xa, xb, np.array([1, 10]))
    # ! Third histogram filled in once we know the worst values
    # ! Fourth histogram filled in once we know optimal values

    # --- (B)
    steps = 720
    thetas = np.linspace(0, np.pi * 2, num=steps)
    test_mat = np.vstack((thetas, np.empty(steps)))
    orig_w = np.array([1, 0])
    max_val = float('-inf')
    w_star = []
    # Test for each theta
    for i, theta in enumerate(test_mat[0]):
        w = calc_R_theta(theta).dot(orig_w)
        ya = calc_y(xa, w)
        yb = calc_y(xb, w)
        test_mat[1][i] = calc_fisher_ratio(ya, yb)
        if (test_mat[1][i] > max_val):
            w_star = w
            max_val = test_mat[1][i]

    # Process test results
    print("[W* = " + str(w_star) + "] [Ratio = " + str(max_val) + "]")
    q1b_fig, q1b_ax = plt.subplots(ncols=1, nrows=1)
    q1b_ax.plot(test_mat[0], test_mat[1], color='black')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Ratio')

    # Add worst histogram solution
    min_theta = test_mat[0][np.argmin(test_mat[1])]
    w_bad = calc_R_theta(min_theta).dot(orig_w)
    comp_hist(q1a_ax[1][0], xa, xb, w_bad, title=(
        'W [Worst]: ' + np.array2string(w_bad)))
    # Add best histogram solution
    max_theta = test_mat[0][np.argmax(test_mat[1])]
    comp_hist(q1a_ax[1][1], xa, xb, w_star, title=(
        'W*: ' + np.array2string(w_star)))

    # ! Question 2
    # --- (A)
    q2a_fig, q2a_ax = plt.subplots(ncols=1, nrows=1)

    # Plot weight line
    # Find two points that fall on line that corresponds with angle
    mean = np.add(mean_a, mean_b) / 2
    x1, y1 = mean[0], mean[1]
    x2, y2 = w_star[0] + x1, w_star[1] + y1
    coefficients = np.polyfit([x1, x2], [y1, y2], 1)
    polynomial = np.poly1d(coefficients)
    # Extrapolate line to fit across the whole dataset
    y_axis = polynomial(x_range)
    # Plot the line
    q2a_ax.plot(x_range, y_axis, color='black', linewidth=2, linestyle='dashed')

    # Find the PDF for each distribution, plot as equi-probable contour lines
    X, Y = np.meshgrid(x_range, y_range)
    X_Y_mesh = np.dstack((X, Y))
    Z_a = mvn_a.pdf(X_Y_mesh)
    cs = q2a_ax.contour(X, Y, Z_a, 15, colors='blue', linestyles='dashed', linewidths=1)
    #labs = q2a_ax.clabel(cs, inline=True, fontsize=10)
    Z_b = mvn_b.pdf(X_Y_mesh)
    cs = q2a_ax.contour(X, Y, Z_b, 15, colors='red', linestyles='dashed', linewidths=1)
    #labs = q2a_ax.clabel(cs, inline=True, fontsize=10)

    # --- (B)
    # Calculate the log odds over the plot area
    Z = calc_log_odds(X[0, :], Y[:, 0], mean_a, mean_b, covar_a, covar_b)
    q2a_ax.contour(X, Y, Z, 0, colors='green')

    # Extract those log odds close to 0
    Xs, Ys = [], []
    tol = 1e-17
    for xi, x in enumerate(X[0, :]):
        for yi, y in enumerate(Y[:, 0]):
            log_odd = np.log(np.abs(Z[yi][xi]))
            if log_odd < tol:
                Xs.append(x)
                Ys.append(y)
    print(len(Xs))
    q2a_ax.set_xlim([np.min(comb_x[0]), np.max(comb_x[0])])
    q2a_ax.set_ylim([np.min(comb_x[1]), np.max(comb_x[1])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s0 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma)

    # --- (C)
    # GOD KNOWS WHAT THIS MEANS

    # Make sure we see any issues before showing figures
    sys.stdout.flush()
    # Show generated figures
    plt.show()
