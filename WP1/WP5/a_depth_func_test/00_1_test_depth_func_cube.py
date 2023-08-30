'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from depth_funcs import depth_ftn_mp, gen_usph_vecs_norm_dist as gen_usph_vecs_mp


# plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'axes.labelsize': 20})


ndims = 2
ei = -1 + (2 * np.random.randn(1000, ndims))


def depth_ftn(x, y, ei):

    mins = x.shape[0] * np.ones((y.shape[0],))  # initial value

    for i in ei:  # iterate over unit vectors
        d = np.dot(i, x.T)  # scalar product

        dy = np.dot(i, y.T)  # scalar product

        # argsort gives the sorting indices then we used it to sort d
        d = d[np.argsort(d)]

        dy_med = np.median(dy)
        dy = ((dy - dy_med) * (1 - (1e-10))) + dy_med

        # find the index of each project y in x to preserve order
        numl = np.searchsorted(d, dy)
        # numl is number of points less then projected y
        numg = d.shape[0] - numl

        # find new min
        mins = np.min(
            np.vstack([mins, np.min(np.vstack([numl, numg]), axis=0)]), 0)

    return mins


plt.ioff()

if __name__ == '__main__':
    # from datetime import datetime
    # from std_logger import StdFileLoggerCtrl

    # save all console activity to out_log_file
    # out_log_file = os.path.join(r'P:\\',
    #                             r'Synchronize',
    #                             r'python_script_logs',
    #                             ('xxx_log_%s.log' %
    #                              datetime.now().strftime('%Y%m%d%H%M%S')))
    # log_link = StdFileLoggerCtrl(out_log_file)
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    n_cpus = 1
    n_vecs = int(1e4)

    os.chdir(main_dir)
    np.random.seed(1)
    unit_vecs = gen_usph_vecs_mp(n_vecs, 2, n_cpus)
    rand_pts = np.random.random(size=(500, 2))
    rand_pts_2 = np.random.normal(size=(100, 2))

    depth_ftn(rand_pts_2, rand_pts_2, ei)
    # s[depth_ftn_mp(rand_pts, rand_pts, unit_vecs, n_cpus) == 1]
    chull_pts = rand_pts_2
    hull = ConvexHull(chull_pts)
    depths = depth_ftn_mp(chull_pts, chull_pts, unit_vecs, n_cpus)
    # np.where(chull_pts==np.median(chull_pts, axis=0))
    # np.where(depths == 115)
    # depths[115]
    # max(depths)
    corner_pts = chull_pts[depths == 1, :]
    depth_pts_2 = chull_pts[depths == 2, :]
    depth_pts_10perc = chull_pts[depths >= np.percentile(depths, 90)]
    depth_pts_50perc = chull_pts[depths >= np.percentile(depths, 50)]
    plt.ioff()
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(chull_pts[:, 0], chull_pts[:, 1],
                #label='Data set',
                alpha=0.50)

    # plt.scatter(depth_pts_2[:, 0], depth_pts_2[:, 1],
    # label='rands (d==2)', alpha=0.7)
    # plt.scatter(depth_pts_50perc[:, 0], depth_pts_50perc[:, 1],
    # label='50perc', alpha=0.7)
    plt.scatter(depth_pts_10perc[:, 0], depth_pts_10perc[:, 1],
                label='Data center',
                alpha=0.7)
    # plt.scatter(unit_vecs[:, 0], unit_vecs[:, 1], alpha=0.7)
    for simplex in hull.simplices:

        plt.plot(chull_pts[simplex, 0],
                 chull_pts[simplex, 1], 'k-',
                 alpha=0.75)

    plt.scatter(corner_pts[:, 0], corner_pts[:, 1],
                label='Depth (d==1)', alpha=0.9, c='lime',
                marker='d')

    plt.axvline(chull_pts[:, 0][4],
                linestyle='--', label='hyperplane',
                c='m', alpha=0.5)

    plt.scatter(chull_pts[:, 0][4], chull_pts[:, 1][4], marker='X',
                s=100, c='r',
                label='d=%d' % depths[
                    np.where(np.abs(chull_pts[:, 0] -
                                    chull_pts[:, 0][4]) < 0.0001)[0]])

    # depths[np.where(np.abs(chull_pts[:,0]-0.64575154)<0.0001)[0]]
    plt.legend(framealpha=0.5, ncol=2)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.show()
    plt.savefig(
        r'X:\staff\elhachem\PhD_Dissertation\thesis-tex-files\Pictures\depth_ex.png')
    # plt.show()
    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    # log_link.stop()
