import numpy as np
from matplotlib import pyplot as plt

N = 100
dim = 30
lamda = 1/np.sqrt(N);
np.random.seed(50)
w = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim))).T
X = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N))
y = X*w

L = (np.linalg.svd(X)[1][0])**2
print(L)
max_iter = 100

def obj(w):
    r = X*w-y;
    return np.sum(np.multiply(r,r))/2 +  lamda * np.sum(np.abs(w))

def subgrad(w):
    return  X.T*(X*w-y) + lamda*np.sign(w) 

def f_grad(w):
    return  X.T*(X*w-y) 

def soft_threshod(w,mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

def smooth_grad(w, mu):
    temp = np.multiply((np.abs(w)<=mu),w/mu) + np.multiply((np.abs(w)>mu),np.sign(w)) 
    return X.T*(X*w-y) + lamda * temp

import cvxpy as cvx

w = cvx.Variable(dim)
loss = cvx.sum_squares(X*w-y)/2 + lamda * cvx.norm(w,1)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))

#Proximal gradient
w = np.matrix([0.0]*dim).T
obj_PG = []
for t in range(0, max_iter):
    obj_val = obj(w)
    w = w - (1/L)* f_grad(w)
    w= soft_threshod(w,lamda/L)
    
    obj_PG.append(obj_val.item())
    if (t%5==0):
        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))

## Nesterovs' Accelerated Proximal Gradient
w = np.matrix([0.0]*dim).T
v = w
obj_APG = []
gamma = 1/L
for t in range(0, max_iter):
    obj_val = obj(w)
    w_prev = w
    w = v - gamma * f_grad(v)
    w = soft_threshod(w,lamda * gamma)
    v = w + t/(t+3) * (w - w_prev)

    obj_APG.append(obj_val.item())
    if (t%5==0):
        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))

## ADMM
w = np.matrix([0.0]*dim).T
z = w
u = w
obj_ADMM = []
rho = 5
for t in range(0, max_iter):
    obj_val = obj(w)
    w = np.linalg.inv((X.T)*X + rho*np.identity(dim))*(X.T*y + rho*z - u )
    z= soft_threshod(w + u/rho,lamda/rho)
    u = u + rho * (w-z)
    obj_ADMM.append(obj_val.item())
    if (t%5==0):
        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))

t = np.arange(0, max_iter)
fig, ax = plt.subplots(figsize = (9, 6))
plt.semilogy(t, np.array(obj_PG) - opt, 'b', linewidth = 2, label = 'Proximal Gradient')
plt.semilogy(t, np.array(obj_APG) - opt, 'c--', linewidth = 2, label = 'Accelerated Proximal Gradient')
plt.semilogy(t, np.array(obj_ADMM) - opt, 'r', linewidth = 2, label = 'ADMM')
plt.legend(prop={'size':12})
plt.xlabel('Iteration')
plt.ylabel('Objective error')
plt.show()

#logistic loss 
from sklearn import datasets
from sklearn.preprocessing import normalize
from scipy.special import expit as sigmoid

N = 10000
dim = 50
lamda = 1e-4
np.random.seed(1)
w = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim))).T
X = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N))
X = np.matrix(normalize(X, axis=1, norm='l2'))
y = 2 * (np.random.uniform(size = (N, 1)) < sigmoid(X*w)) - 1

w = cvx.Variable(dim)
loss = 1/N * cvx.sum_entries(cvx.logistic(-cvx.mul_elemwise(y, X*w))) + lamda/2 * cvx.sum_squares(w)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True, abstol=1e-15) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))