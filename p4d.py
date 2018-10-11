import numpy as np
import numpy.linalg as la
from svd_solve import svd, svd_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_plane_LSE import fit_plane_LSE, fit_plane_LSE_RANSAC

points_file = open('clean_hallway.txt', 'r')
#points_file = open('cluttered_hallway.txt', 'r')
points_lines = points_file.readlines()
points_array = [[float(s) for s in line.strip().split(' ')] for line in points_lines]
points = np.array(points_array, dtype=np.float32)
# homogeneous coord
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

p_set_1 = points
p1, inlier_list1, outlier_list1 = fit_plane_LSE_RANSAC(p_set_1, return_outlier_list=True)
p_set_2 = p_set_1[outlier_list1, :]
p2, inlier_list2, outlier_list2 = fit_plane_LSE_RANSAC(p_set_2, return_outlier_list=True)
p_set_3 = p_set_2[outlier_list2, :]
p3, inlier_list3, outlier_list3 = fit_plane_LSE_RANSAC(p_set_3, return_outlier_list=True)
p_set_4 = p_set_3[outlier_list3, :]
p4, inlier_list4, outlier_list4 = fit_plane_LSE_RANSAC(p_set_4, return_outlier_list=True)

'''
x = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), 0.1)
z = np.arange(np.min(points[:, 2]), np.max(points[:, 2]), 0.1)

xx, zz = np.meshgrid(x, z)

yy = (-p[0]*xx -p[2]*zz -p[3])/p[1]
'''

def draw_plane(p, points, ax):
    normal = np.array([p[0], p[1], p[2]])
    normal_heading_axis = np.argmax(np.abs(normal))
    if normal_heading_axis == 0:
        # grid with yz
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        
        y = np.arange(y_min, y_max, 0.1)
        z = np.arange(z_min, z_max, 0.1)
        yy, zz = np.meshgrid(y, z)
        xx = (-p[1]*yy - p[2]*zz - p[3])/p[0]
        ax.plot_surface(xx, yy, zz)
    elif normal_heading_axis == 1:
        # grid with xz
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        
        x = np.arange(x_min, x_max, 0.1)
        z = np.arange(z_min, z_max, 0.1)
        xx, zz = np.meshgrid(x, z)
        yy = (-p[0]*xx - p[2]*zz - p[3])/p[1]
        ax.plot_surface(xx, yy, zz)
    elif normal_heading_axis == 2:
        # grid with xy
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        xx, yy = np.meshgrid(x, y)
        zz = (-p[0]*xx - p[1]*yy - p[3])/p[2]
        ax.plot_surface(xx, yy, zz)

def draw_points(points, ax, c='r'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rand_choice = np.random.choice(points.shape[0], 10000, replace=False)
ax.scatter(points[rand_choice, :][:, 0], points[rand_choice, :][:, 1], points[rand_choice, :][:, 2], c='r')



draw_plane(p1, points, ax)
draw_plane(p2, points, ax)
draw_plane(p3, points, ax)
draw_plane(p4, points, ax)


'''
draw_points(p_set_1[inlier_list1, :], ax, 'r')
draw_points(p_set_2[inlier_list2, :], ax, 'g')
draw_points(p_set_3[inlier_list3, :], ax, 'b')
draw_points(p_set_4[inlier_list4, :], ax, 'y')
'''

plt.show()