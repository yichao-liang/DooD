import numpy as np
from scipy.interpolate import BSpline
import skimage
import matplotlib.pyplot as plt

def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))

def scale(x, size):                                                                                                                                                                                                                                                    
    s = size / (x.max() - x.min())                                                                                                                                                                                                                                     
    return ((x - x.min())*s).astype(np.int16)

k = 2
t = [0, 1, 2, 3, 4, 5, 6]
c = [-1, 2, 0, -1]
spl = BSpline(t, c, k)
spl(2.5)
bspline(2.5, t, c, k)

size = 20
img = np.zeros((size, size), dtype=np.uint8)                                                                                                                                                                                                                               

xx = np.linspace(1.5, 4.5, 50)
x_ = size - scale(xx, size - 1) - 1
y_ = size - scale(spl(xx), size - 1) - 1
img[y_, x_] = 1


for i in range(20):
    for j in range(20):
        if img[i, j] == 1:
            print("# ", end='')
        else:
            print(". ", end='')
    print()

print(img)

plt.imshow(img)
plt.show()
