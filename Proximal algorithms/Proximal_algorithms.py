import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt

def least_squares(x, features, labels):
    n_samples = features.shape[0]
    x = x.reshape(1, n_features)
    loss_array = (features.dot(x.T) - labels) ** 2
    return np.sum(loss_array, axis=0) / (2. * n_samples)

def least_squares_grad(x, features, labels):
    n_samples = features.shape[0]
    x = x.reshape(1, n_features)
    grad_array = (features.dot(x.T) - labels)* features
    return np.sum(grad_array, axis=0) / n_samples

logistic = lambda x: 1. / (1. + np.exp(-x))

def logistic_loss(x, features, labels):
    n_samples, n_features = features.shape
    x = x.reshape(1, n_features)
    loss_array = np.log(1 + np.exp(-labels * features.dot(x.T)))
    return np.sum(loss_array, axis = 0)/ n_samples

def logistic_loss_grad(x, features, labels):
    n_samples = features.shape[0]
    x = x.reshape(1, n_features)
    grad_array = -labels / (1 + np.exp(labels * features.dot(x.T))) * features
    return np.sum(grad_array, axis=0) / n_samples

def prox_l1(x, l=1.):
    x_abs = np.abs(x)
    return np.sign(x) * (x_abs - 1)*(x_abs > 1)

def prox_l2(x, l=1.):
    return 1 / (1+l) / x

def prox_enet(x, l_l1, l_l2, t=1.):
    x_abs = np.abs(x)
    prox_l1 = np.sign(x) * (x_abs - t * l_l1) * (x_abs > t * l_l1)
    return prox_l1 / (1. + t * l_l2)

def inspector(loss_fun, x_real, verbose = False):
    objectives = []
    errors = []
    it = [0]
    def inspector_cl(xk):
        obj = loss_fun(xk)
        err = norm(xk - x_real) / norm(x_real)
        objectives.append(obj)
        errors.append(err)
        if verbose == True:
            if it[0] == 0:
                print (' | '.join([name.center(8) for name in ["it", "obj", "err"]]))
            if it[0] % (n_iter / 5) == 0:
                print (' | '.join([("%d" % it[0]).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % err).rjust(8)]))
            it[0] += 1
    inspector_cl.obj = objectives
    inspector_cl.err = errors
    return inspector_cl

def gd(x_init, grad, n_iter=100, step=1., callback=None):
    """Basic gradient descent algorithm."""
    x = x_init.copy()
    
    for _ in range(n_iter):
        x -= step * grad(x)
        
        # Update metrics after each iteration.
        if callback is not None:
            callback(x)
    return x

def ista(x_init, grad, prox, n_iter = 100, step = 1., callback = None):
    x = x_init.copy()

    for _ in range(n_iter):
        x = prox(x - step *grad(x), step)

        if callback is not None:
            callback(x)
    
    return x

def fista(x_init, grad, prox, n_iter=100, step=1., callback=None):
    """FISTA algorithm."""
    x = x_init.copy()
    y = x_init.copy()
    t = 1.
    
    for _ in range(n_iter):
        x_new = prox(y - step * grad(y), step)
        t_new = (1. + (1. + 4. * t**2)**.5) / 2
        y = x_new + (t - 1) / t_new * (x_new - x)
        t = t_new
        x = x_new

        # Update metrics after each iteration.
        if callback is not None:
            callback(x)
    return x

#Generate a fake dataset

n_samples = 2000
n_features = 50

idx = np.arange(n_features).reshape(1, n_features)
params = 2 * (-1) ** (idx - 1) * .9**idx
params[0, 20:50] = 0
diag = np.random.rand(n_features)
features = np.random.multivariate_normal(np.zeros(n_features), np.diag(diag), n_samples)

# Show the condition number of the gram matrix
print("cond = %.2f" % (diag.max() / diag.min()))

linear = True
if linear == True:
    residuals = np.random.randn(n_samples, 1)
    labels = features.dot(params.T) + residuals
else:
     labels = np.array([[float(np.random.rand() < p)] for p in logistic(features.dot(params.T))])

plt.figure(figsize=(8, 4))
plt.stem(params[0])
plt.title("True parameters", fontsize=16)
plt.show()

x_init = 1 - 2 * np.random.rand(1, n_features)
n_iter = 30
l_l1 = 0.0
l_l2 = 0.1

#f and gradient
if linear == True:
    f = lambda x: least_squares(x, features, labels)
    grad_f = lambda x: least_squares_grad(x, features, labels)
    step = norm(features.T.dot(features)/ n_samples, 2)
else:
    f = lambda x: logistic_loss(x, features, labels)
    grad_f = lambda x: logistic_loss_grad(x, features, labels)
    step = 1.

#g, F and prox
g = lambda x: l_l1 * np.abs(x).sum() + 0.5 * l_l2 * np.sum(x**2)
F = lambda x: f(x) + g(x)
prox_g = lambda x, l: prox_enet(x, l_l1, l_l2, l)

print("Type: %s" % ('linear' if linear else 'logistic'))
print("n_iter: %d" % n_iter)
print("step size: %.2f" % step)

import scipy.optimize

ls = lambda x: logistic_loss(x, features, labels)
print(scipy.optimize.approx_fprime(x_init.ravel(), ls, 1e-3))
print(logistic_loss_grad(x_init, features, labels))

plt.figure(figsize=(8,4))
plt.stem(x_init[0])
plt.title("Initial guess", fontsize=16)
plt.show()

#ISTA
ista_inspector = inspector(loss_fun=F, x_real = params, verbose = True)
x_ista = ista(x_init, grad=grad_f, prox=prox_g, n_iter=n_iter, step=step, callback=ista_inspector)

#FISTA
fista_inspector = inspector(loss_fun=F, x_real=params, verbose=True)
x_fista = fista(x_init, grad=grad_f, prox=prox_g, n_iter=n_iter, step=step, callback=fista_inspector)

#Gradient descent
grad_gd = lambda x: grad_f(x) + l_l1 * np.abs(x) + l_l2 * x
gd_inspector = inspector(loss_fun=F, x_real=params, verbose=True)
x_gd = gd(x_init, grad=grad_gd, n_iter=n_iter, step=step, callback=gd_inspector)

plt.figure(figsize=(18, 5))
plt.suptitle("Final estimates", fontsize=18)
plt.subplot(1, 4, 1)
plt.title("Real params")
plt.stem(params[0])
plt.subplot(1, 4, 2)
plt.title("ISTA")
plt.stem(x_ista[0], color='red')
plt.subplot(1, 4, 3)
plt.title("FISTA")
plt.stem(x_fista[0])
plt.subplot(1, 4, 4)
plt.title("GD")
plt.stem(x_gd[0])
plt.show()

plt.figure(figsize=(17, 5))
plt.subplot(1, 2, 1)
plt.plot(gd_inspector.obj, 'b')
plt.plot(ista_inspector.obj, 'r')
plt.plot(fista_inspector.obj, 'g')
plt.title("Loss", fontsize=18)
plt.xlabel("iteration", fontsize=14)
plt.ylabel("value", fontsize=14)
plt.legend(["gd", "ista", "fista"])
plt.subplot(1, 2, 2)
plt.plot(gd_inspector.err, 'b')
plt.plot(ista_inspector.err, 'r')
plt.plot(fista_inspector.err, 'g')
plt.title("Distance to x_real", fontsize=18)
plt.xlabel("iteration", fontsize=14)
plt.ylabel("distance", fontsize=14)
plt.legend(["gd", "ista", "fista"])
plt.show()