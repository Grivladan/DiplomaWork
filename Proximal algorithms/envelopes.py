import numpy as np
import math
import matplotlib.pylab as plt

np.random.seed(0)
N = 400
P = int(round(N/4))
A = np.random.randn(P,N) / math.sqrt(P)

S = 17
sel = np.random.permutation(N)
sel = sel[0:S]   # indices of the nonzero elements of xsharp
xsharp = np.zeros(N)
xsharp[sel] = 1

y = A.dot(xsharp)

def prox_gamma_g (x, gamma) :
    return x - x/np.maximum(abs(x)/gamma,1) # soft-thresholding

plt.figure(figsize=(9, 6))
t = np.arange(-1,1,0.001)
plt.plot(t, prox_gamma_g(t,0.3))
plt.axis('equal')
plt.show()

pA = np.linalg.pinv(A) # pseudo-inverse. Equivalent to pA = A.T.dot(inv(A.dot(A.T)))
def prox_f (x, y) :
    return x + pA.dot(y-A.dot(x))

gamma = 0.1 # try 1, 10, 0.1
rho = 1     # try 1, 1.5, 1.9
nbiter = 700

s = np.zeros(N)
En_array = np.zeros(nbiter)
for iter in range(nbiter):  # iter goes from 0 to nbiter-1
    x = prox_f(s, y)
    s = s + rho * (prox_gamma_g(2*x - s, gamma) - x)
    En_array[iter] = np.linalg.norm(x, ord=1)  
x_restored = x

fig, (subfig1,subfig2) = plt.subplots(1,2,figsize=(16,7)) # one figure with two horizontal subfigures
subfig1.stem(xsharp)
subfig1.set_ylim(0,1.1)
subfig2.stem(x_restored)
subfig2.set_ylim(0,1.1)
subfig1.set_title('$x^\sharp$')
subfig2.set_title('$x_\mathrm{restored}$')
plt.show()

plt.plot(np.log10(En_array-En_array.min()))
plt.show()