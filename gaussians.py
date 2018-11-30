import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sps
from mpl_toolkits.mplot3d import Axes3D


def get_test_classes():
    cs = []
    # Main test classes
    cs.append(mtc([5, 5], [[3, 1], [1, 1]], 1000, 'red'))
    cs.append(mtc([1, 1], [[2, 1], [1, 2]], 1000, 'blue'))
    # Test class [0] with variance of [1]
    cs.append(mtc([5, 5], [[2, 1], [1, 2]], 1000, 'lawngreen'))
    # Test class [1] with less points
    cs.append(mtc([1, 1], [[2, 1], [1, 2]], 100, 'yellow'))
    # Class with no x-y variance correlation
    cs.append(mtc([1, 5], [[1, 0], [0, 1]], 1000, 'purple'))
    return cs


def mtc(mean, covar, points, color):
    # Make test class
    mean = np.array(mean)
    covar = np.array(covar)
    data, mvn = get_normal_data(mean, covar, points)
    clazz = {'x': data, 'mean': mean, 'covar': covar,
             'points': points, 'mvn': mvn, 'color': color}
    return clazz


def get_normal_data(mean, covar, points):
    mvn = sps.multivariate_normal(mean=mean, cov=covar)
    data = mvn.rvs(size=points)
    sys.stderr.flush()
    return np.array(data).T, mvn


def comp_hist(ax, xa, xb, w, color_a, color_b, **kwargs):
    ya = calc_y(xa, w)
    yb = calc_y(xb, w)
    # Plot histogram
    ax.hist(ya, alpha=0.5, color=color_a, density=True)
    ax.hist(yb, alpha=0.5, color=color_b, density=True)
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
    count_a, count_b = np.size(ya), np.size(yb)
    top = (np.mean(ya) - np.mean(yb)) ** 2
    bot_lhs = count_a / (count_a + count_b) * np.var(ya)
    bot_rhs = count_b / (count_a + count_b) * np.var(yb)
    ratio = top / (bot_lhs + bot_rhs)
    return ratio


def calc_unbalanced_discriminant(ya, yb):
    count_a, count_b = np.size(ya), np.size(yb)
    top = (np.mean(ya) - np.mean(yb)) ** 2
    bot = np.var(ya) + np.var(yb)
    ratio = top / bot
    return ratio


def calc_log_odds(xs, ys, mean_a, mean_b, covar_a, covar_b):
    Lam_a = np.linalg.inv(covar_a)
    Lam_b = np.linalg.inv(covar_b)
    lhs = np.log(np.linalg.det(Lam_a) / np.linalg.det(Lam_b))
    Z = np.empty((len(xs), len(ys)))
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            p = np.array([x, y])
            g_a = (p - mean_a).T.dot(Lam_a).dot(p - mean_a) / 2
            g_b = (p - mean_b).T.dot(Lam_b).dot(p - mean_b) / 2
            Z[yi][xi] = lhs + g_b - g_a
    return Z


def process_datasets(class_a, class_b):
    # Dataset A
    count_a = class_a['points']
    mean_a = class_a['mean']
    covar_a = class_a['covar']
    xa = class_a['x']
    mvn_a = class_a['mvn']
    color_a = class_a['color']

    # Dataset B
    count_b = class_b['points']
    mean_b = class_b['mean']
    covar_b = class_b['covar']
    xb = class_b['x']
    mvn_b = class_b['mvn']
    color_b = class_b['color']

    # Combined Dataset
    comb_x = np.hstack((xa, xb))

    # --- Data plot of dataset
    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    data_ax.scatter(xa[0], xa[1], color=color_a, alpha=0.5)
    data_ax.scatter(xb[0], xb[1], color=color_b, alpha=0.5)

    # ! Question 1
    # --- (A)
    test_proj_fig, test_proj_ax = plt.subplots(ncols=2, nrows=1)
    comp_hist(test_proj_ax[0], xa, xb, np.array([1, 1]), color_a, color_b)
    comp_hist(test_proj_ax[1], xa, xb, np.array([1, 10]), color_a, color_b)

    # --- (B)
    steps = 720
    thetas = np.linspace(0, np.pi * 2, num=steps)
    test_mat = np.vstack((thetas, np.empty(steps)))
    orig_w = np.array([1, 0])
    max_val, min_val = float('-inf'), float('inf')
    w_star, w_worst = [], []
    # Test the fisher ratio for each theta
    for i, theta in enumerate(test_mat[0]):
        w = calc_R_theta(theta).dot(orig_w)
        ya = calc_y(xa, w)
        yb = calc_y(xb, w)
        test_mat[1][i] = calc_fisher_ratio(ya, yb)
        if (test_mat[1][i] > max_val):
            w_star = w
            max_val = test_mat[1][i]
        if (test_mat[1][i] < min_val):
            w_worst = w
            min_val = test_mat[1][i]

    # Process test results
    print("[W* = " + str(w_star) + "] [Ratio = " + str(max_val) + "]")
    # Plot theta results
    theta_fig, theta_ax = plt.subplots(ncols=1, nrows=1)
    theta_ax.plot(test_mat[0], test_mat[1],
                  color='black', label='Fisher Ratio')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Ratio')

    # Plot projection results for best and worst thetas
    found_proj_fig, found_proj_ax = plt.subplots(ncols=2, nrows=1)
    # Add worst histogram solution
    comp_hist(found_proj_ax[0], xa, xb, w_worst, color_a, color_b,
              title=('W [Worst]: ' + np.array2string(w_worst)))
    # Add best histogram solution
    comp_hist(found_proj_ax[1], xa, xb, w_star, color_a, color_b,
              title=('W*: ' + np.array2string(w_star)))

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
    q2a_ax.plot(x_range, y_axis, color='black',
                linewidth=2, linestyle='dashed')

    # Find the PDF for each distribution, plot as equi-probable contour lines
    X, Y = np.meshgrid(x_range, y_range)
    X_Y_mesh = np.dstack((X, Y))
    Z_a = mvn_a.pdf(X_Y_mesh)
    cs = q2a_ax.contour(X, Y, Z_a, 15, colors=color_a,
                        linestyles='dashed', linewidths=1)
    #labs = q2a_ax.clabel(cs, inline=True, fontsize=10)
    Z_b = mvn_b.pdf(X_Y_mesh)
    cs = q2a_ax.contour(X, Y, Z_b, 15, colors=color_b,
                        linestyles='dashed', linewidths=1)
    #labs = q2a_ax.clabel(cs, inline=True, fontsize=10)

    # --- (B)
    # Calculate the log odds over the plot area
    Z = calc_log_odds(X[0, :], Y[:, 0], mean_a, mean_b, covar_a, covar_b)
    q2a_ax.contour(X, Y, Z, 0, colors='green')
    # Constrain the axis to where we expect data
    q2a_ax.set_xlim([np.min(comb_x[0]), np.max(comb_x[0])])
    q2a_ax.set_ylim([np.min(comb_x[1]), np.max(comb_x[1])])

    # Make a 3D plot of the log ratios for reference
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s0 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma)

    # --- (C)
    # Create another copy of the graph from q1b so we can compare the discriminant
    q2c_fig, q2c_ax = plt.subplots(ncols=1, nrows=1)
    q2c_ax.plot(test_mat[0], test_mat[1], color='black', label='Fisher Ratio')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Ratio')

    steps = 720
    thetas = np.linspace(0, np.pi * 2, num=steps)
    test_mat = np.vstack((thetas, np.empty(steps)))
    orig_w = np.array([1, 0])
    max_val = float('-inf')
    w_star = []
    # Test the discriminant for each theta
    for i, theta in enumerate(test_mat[0]):
        w = calc_R_theta(theta).dot(orig_w)
        ya = calc_y(xa, w)
        yb = calc_y(xb, w)
        test_mat[1][i] = calc_unbalanced_discriminant(ya, yb)
        if (test_mat[1][i] > max_val):
            w_star = w
            max_val = test_mat[1][i]
    q2c_ax.plot(test_mat[0], test_mat[1], color='red', label='Unbalanced')
    q2c_ax.legend()


def plot_classes_data(classes):
    data_fig, data_ax = plt.subplots(ncols=1, nrows=1)
    contour_fig, contour_ax = plt.subplots(ncols=1, nrows=1)
    for ci, clazz in enumerate(classes):
        data_ax.scatter(clazz['x'][0], clazz['x'][1], color=clazz['color'],
                        alpha=0.50, label=('Class ' + str(ci)))
        X, Y = np.meshgrid(x_range, y_range)
        X_Y_mesh = np.dstack((X, Y))
        Z = clazz['mvn'].pdf(X_Y_mesh)
        cs = contour_ax.contour(X, Y, Z, 15, colors=clazz['color'],
                                linestyles='dashed', linewidths=2)

    data_ax.legend()
    contour_ax.set_xlim(data_ax.get_xlim())
    contour_ax.set_ylim(data_ax.get_ylim())


####################################################
# MAIN
####################################################
if __name__ == "__main__":
    # Set the seed so results are repeatable
    np.random.seed(0)
    # Configure the test space
    x_range = np.linspace(-10, 20, 100)
    y_range = np.linspace(-10, 20, 100)

    # Get a copy of all classes
    classes = get_test_classes()

    # --- Data plot of dataset
    plot_classes_data(classes)

    # Do comparison of two test classes
    class_a, class_b = 1, 2
    process_datasets(classes[class_a], classes[class_b])

    # Make sure we see any issues before showing figures
    sys.stdout.flush()
    sys.stderr.flush()
    # Show generated figures
    plt.show()
