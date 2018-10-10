import numpy as np
import numpy.linalg as la
from svd_solve import svd, svd_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_plane_LSE import fit_plane_LSE, fit_plane_LSE_RANSAC

points_file = open('cluttered_table.txt', 'r')
points_lines = points_file.readlines()
points_array = [[float(s) for s in line.strip().split(' ')] for line in points_lines]
points = np.array(points_array, dtype=np.float32)
# homogeneous coord
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


p, inlier_list = fit_plane_LSE_RANSAC(points)

x = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), 0.1)
z = np.arange(np.min(points[:, 2]), np.max(points[:, 2]), 0.1)

xx, zz = np.meshgrid(x, z)

yy = (-p[0]*xx -p[2]*zz -p[3])/p[1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')
ax.plot_wireframe(xx, yy, zz)

plt.show()