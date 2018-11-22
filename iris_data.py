import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as spy
import scipy.linalg

####################################################
# MAIN
####################################################
if __name__ == "__main__":
    header = ['sl', 'sw', 'pl', 'pw', 'class']
    df = pd.read_csv('data/iris.data', header=None, names=header)
    xs = df[['sl', 'sw', 'pl', 'pw']]
    y = df['class']

    # Allocate a colour for each class
    classes = np.unique(y).tolist()
    class_cnt = len(classes)
    colors = ['red', 'green', 'blue', 'purple', 'yellow']
    class_colors = [colors[classes.index(c)] for c in df['class']]
    # Plot some data graphs
    data_fig, data_ax = plt.subplots(ncols=2, nrows=1)
    for ci, c in enumerate(classes):
        cd = xs.loc[y == c]
        kwargs = {'alpha': 0.5, 'color': colors[ci], 'label': c}
        data_ax[0].scatter(cd['sl'], cd['sw'], **kwargs)
        data_ax[1].scatter(cd['pl'], cd['pw'], **kwargs)
    # Configure graph
    data_ax[0].set_xlabel('Sepal Length (cm)')
    data_ax[0].set_ylabel('Sepal Width (cm)')
    data_ax[1].set_xlabel('Petal Length (cm)')
    data_ax[1].set_ylabel('Petal Width (cm)')
    data_fig.legend(*data_ax[0].get_legend_handles_labels())

    # ! Section 3
    # --- Question 1
    mean = np.array(np.mean(xs)).reshape(class_cnt + 1, 1)
    # Calculate covariance matrices
    covar_b, covar_w = 0, 0
    for ci, c in enumerate(classes):
        mean_c = np.mean(xs.loc[y == c])
        mean_c = np.array(mean_c).reshape(class_cnt + 1, 1)

        # Calculate for within class covariance matrix
        cd = np.array(xs.loc[y == c])
        vals = cd.shape[0]
        for x in cd:
            x = x.reshape(class_cnt + 1, 1)
            covar_w += (x - mean_c).dot((x - mean_c).T)
        # Calculate for between class covariance matrix (with normalisation)
        covar_b += vals * (mean_c - mean).dot((mean_c - mean).T)
    #print("Covariance Between:")
    # print(covar_b)
    #print("Covariance Within:")
    # print(covar_w)

    # Calculate generalised eigenvalues/eigenvectors
    eig_vals, eig_vecs = spy.linalg.eigh(covar_b, covar_w)
    eig_val, eig_vec = eig_vals[-1], eig_vecs[:, -1]
    # Verify that the generalised eigenvector condition holds
    eig_condition = (covar_b - eig_val * covar_w) @ eig_vec
    np.testing.assert_array_almost_equal(eig_condition, np.zeros(class_cnt+1))

    # Get the largest eigenvalue and its corresponding eigenvector
    w_star = eig_vec
    xs_wstar= np.array(xs) @ w_star

    # --- Question 2
    w_star_fig, w_star_ax = plt.subplots(ncols=1, nrows=1)   
    for ci, c in enumerate(classes):
        w_star_ax.hist(xs_wstar[y == c], alpha=0.5, color=colors[ci])

    # --- Question 3
    q3_fig, q3_ax = plt.subplots(ncols=1, nrows=class_cnt)
    q3_ax[class_cnt // 2].set_ylabel('Number of Values')
    q3_ax[class_cnt - 1].set_xlabel('Projected Value')   
    for i, translation in enumerate(eig_vecs[:-1]):
        w = w_star - translation*1000
        xs_w = np.array(xs) @ w
        for ci, c in enumerate(classes):
            q3_ax[i].hist(xs_w[y == c], alpha=0.5, color=colors[ci], density=True)

    print(eig_vecs)
    # Make sure we see any issues before showing figures
    sys.stdout.flush()
    sys.stderr.flush()
    # Show generated figures
    plt.show()
