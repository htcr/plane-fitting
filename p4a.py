import numpy as np
import numpy.linalg as la
from svd_solve import svd, svd_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_plane_LSE import fit_plane_LSE

clear_table_file = open('clear_table.txt', 'r')
#clear_table_file = open('cluttered_table.txt', 'r')
clear_table_lines = clear_table_file.readlines()
clear_table_array = [[float(s) for s in line.strip().split(' ')] for line in clear_table_lines]
clear_table = np.array(clear_table_array, dtype=np.float32)
# homogeneous coord
clear_table = np.concatenate((clear_table, np.ones((clear_table.shape[0], 1))), axis=1)

A = clear_table

p = fit_plane_LSE(A)

# calculate error:
dists = np.abs(A @ p) / np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
avg_dist = np.mean(dists)
print(avg_dist)


x = np.arange(np.min(clear_table[:, 0]), np.max(clear_table[:, 0]), 0.1)
z = np.arange(np.min(clear_table[:, 2]), np.max(clear_table[:, 2]), 0.1)

xx, zz = np.meshgrid(x, z)

yy = (-p[0]*xx -p[2]*zz -p[3])/p[1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(clear_table[:, 0], clear_table[:, 1], clear_table[:, 2], c='r')
ax.plot_wireframe(xx, yy, zz)

plt.show()