import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
a = (1, 3, 5)
b = (4, 2, 9)
ax.plot(a, b, 'ro')
ax.legend()

plt.show()