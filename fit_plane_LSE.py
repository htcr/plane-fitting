import numpy as np
import numpy.linalg as la
from svd_solve import svd, svd_solve

def fit_plane_LSE(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    # ax+by+cz+d = 0
    assert points.shape[0] >= 3 # at least 3 points needed
    U, S, Vt = svd(points)
    null_space = Vt[-1, :]
    return null_space

def get_point_dist(points, plane):
    # return: 1d array of size N (number of points)
    dists = np.abs(points @ plane) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    return dists

def fit_plane_LSE_RANSAC(points, iters=100, inlier_thresh=0.005):
    # points: Nx4 homogeneous 3d points
    # return: 
    #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
    #   inlier_list: 1d array of size N of inlier points
    max_inlier_num = -1
    max_inlier_list = None
    
    N = points.shape[0]
    assert N >= 3

    for i in range(iters):
        chose_id = np.random.choice(N, 3, replace=False)
        chose_points = points[chose_id, :]
        tmp_plane = fit_plane_LSE(chose_points)
        
        dists = get_point_dist(points, tmp_plane)
        tmp_inlier_list = np.where(dists < inlier_thresh)[0]
        tmp_inliers = points[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list
        
        print('iter %d, %d inliers' % (i, max_inlier_num))

    final_points = points[max_inlier_list, :]
    plane = fit_plane_LSE(final_points)
    dists = get_point_dist(points, plane)
    inlier_list = np.where(dists < inlier_thresh)[0]
    return plane, inlier_list

    
    