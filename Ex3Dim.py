from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors


w = np.array([1., 1., 1.])


halfspaces = np.array([
                    [1.*w[0], 1.*w[1], 1.*w[2], -10 ],
                    [ 1.,  0.,  0., -4],
                    [ 0.,  1.,  0., -4],
                    [ 0.,  0.,  1., -4],
                    [-1.,  0.,  0.,  0],
                    [ 0., -1.,  0.,  0],
                    [ 0.,  0., -1.,  0]
                    ])
feasible_point = np.array([0.1, 0.1, 0.1])
hs = HalfspaceIntersection(halfspaces, feasible_point)
verts = hs.intersections
hull = ConvexHull(verts)
faces = hull.simplices

ax = a3.Axes3D(plt.figure())
ax.dist=10
ax.azim=30
ax.elev=10
ax.set_xlim([0,5])
ax.set_ylim([0,5])
ax.set_zlim([0,5])


ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2], triangles=faces, alpha=0.8,color='blue')


plt.show()