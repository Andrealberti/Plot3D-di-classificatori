from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [0, 2, 1, 1]
y = [0, 0, 1, 0]
z = [0, 0, 0, 1]

vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

tupleList = list(zip(x, y, z))

poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
ax.scatter(x,y,z)
ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, alpha=0.5))
ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.2, linestyles=':'))

plt.show()






"""
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3

w = np.array([1., 1., 1.])
# ∑ᵢ hᵢ wᵢ qᵢ - ∑ᵢ gᵢ wᵢ <= 0 
#  qᵢ - ubᵢ <= 0
# -qᵢ + lbᵢ <= 0 
halfspaces = np.array([
                    [1.*w[0], 1.*w[1], 1.*w[2], -10 ],
                    [ 1.,  0.,  0., -4],
                    [ 0.,  1.,  0., -4],
                    [ 0.,  0.,  1., -4],
                    [-1.,  0.,  0.,  0],
                    [ 0., -1.,  0.,  0],
                    [ 0.,  0., -1.,  0]
                    ])

print(halfspaces)
feasible_point = np.array([0.1, 0.1, 0.1])
hs = HalfspaceIntersection(halfspaces, feasible_point)
verts = hs.intersections
hull = ConvexHull(verts)
simplices = hull.simplices

org_triangles = [verts[s] for s in simplices]

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i: 
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups


f = Faces(org_triangles)
g = f.simplify()

ax = a3.Axes3D(plt.figure())

colors = list(map("C{}".format, range(len(g))))

pc = a3.art3d.Poly3DCollection(g,  facecolor=colors, 
                                   edgecolor="k", alpha=0.9)
ax.add_collection3d(pc)

ax.dist=10
ax.azim=30
ax.elev=10
ax.set_xlim([0,5])
ax.set_ylim([0,5])
ax.set_zlim([0,5])
plt.show()



"""