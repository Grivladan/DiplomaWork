import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cvx
import math

N = 100
dim = 30
lamda = 1/np.sqrt(N);
np.random.seed(50)
w = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim))).T
X = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N))
y = X*w

L = (np.linalg.svd(X)[1][0])**2
max_iter = 100

def obj(w):
    r = X*w-y;
    return np.sum(np.multiply(r,r))/2 +  lamda * np.sum(np.abs(w))

def subgrad(X, y, w):
    return  X.T*(X*w-y) + lamda*np.sign(w) 

def f_grad(X, y, w):
    return  X.T*(X*w-y) 

def soft_threshod(w,mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

def hess(X):
    return 2*X.T*X

w = cvx.Variable(dim)
loss = cvx.sum_squares(X*w-y)/2 + lamda * cvx.norm(w,1)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))

#Proximal gradient
def proximal_grad(A,y, f_grad, prox):
    w = np.matrix([0.0]*dim).T
    obj_PG = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - (1/L)* f_grad(A, y, w)
        w= prox(w,lamda/L)
    
        obj_PG.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_PG

#Proximal Newton
def proximal_newton(A, y, f_grad, f_hess, prox):
    w = np.matrix([0.0]*dim).T
    obj_PN = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - (1/L)* np.linalg.inv(f_hess(A))*f_grad(A, y, w)
        w= prox(w,lamda/L)
    
        obj_PN.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_PN

#ISTA
def iterative_shrinkage(A, y, f_grad, prox):
    w = np.matrix([0.0]*dim).T
    obj_ISTA = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - (1/L)* A.T*(A*w - y)
        w= prox(w,lamda/L)
    
        obj_ISTA.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_ISTA


# Nesterovs' Accelerated Proximal Gradient
def accelerated_proximal_gradient(A, y, f_grad, prox):
    w = np.matrix([0.0]*dim).T
    v = w
    obj_APG = []
    gamma = 1/L
    for t in range(0, max_iter):
        obj_val = obj(w)
        w_prev = w
        w = v - gamma * f_grad(A,y,v)
        w = soft_threshod(w,lamda * gamma)
        v = w + t/(t+3.0) * (w - w_prev)

        obj_APG.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_APG

# ADMM
def ADMM(A, y, f_grad, prox):
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
    return obj_ADMM

print('Proximal gradient')
obj_PG = proximal_grad(X, y, f_grad, soft_threshod)

print('Accelerated proximal gradient')
obj_APG = accelerated_proximal_gradient(X, y, f_grad, soft_threshod)

"""
print('Proximal Newton')
obj_APG = proximal_newton(X, y, f_grad, hess, soft_threshod)
"""

print('Ista algorithm')
obj_ISTA = iterative_shrinkage(X, y, f_grad, soft_threshod)
 
print('Alternating direction multipliers method')
obj_ADMM = ADMM(X, y, f_grad, soft_threshod)

t = np.arange(0, max_iter)
fig, ax = plt.subplots(figsize = (9, 6))
plt.semilogy(t, np.array(obj_PG) - opt, 'b', linewidth = 2, label = 'Proximal Gradient')
plt.semilogy(t, np.array(obj_APG) - opt, 'c--', linewidth = 2, label = 'Accelerated Proximal Gradient')
plt.semilogy(t, np.array(obj_ISTA) - opt, 'g', linewidth = 2, label = 'ISTA')
plt.semilogy(t, np.array(obj_ADMM) - opt, 'r', linewidth = 2, label = 'ADMM')
plt.legend(prop={'size':12})
plt.xlabel('Iteration')
plt.ylabel('Objective error')
plt.show()

##logistic loss 
#from sklearn import datasets
#from sklearn.preprocessing import normalize
#from scipy.special import expit as sigmoid

#N = 10000
#dim = 50
#lamda = 1e-4
#np.random.seed(1)
#w = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim))).T
#X = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N))
#X = np.matrix(normalize(X, axis=1, norm='l2'))
#y = 2 * (np.random.uniform(size = (N, 1)) < sigmoid(X*w)) - 1

#w = cvx.Variable(dim)
#loss = 1.0 / N * cvx.sum_entries(cvx.logistic(-cvx.mul_elemwise(y, X*w))) + lamda/2 * cvx.sum_squares(w)

#problem = cvx.Problem(cvx.Minimize(loss))
#problem.solve(verbose=True, abstol=1e-15) 
#opt = problem.value
#print('Optimal Objective function value is: {}'.format(opt))

#L = lamda + 1.0/4;
#num_pass = 50

### Define the objective and gradient oracles. 
#def obj(w):
#    return 1.0/N * np.sum( np.log(1 + np.exp(-np.multiply(y, (X*w)))) ) + 1.0/2 * lamda * (w.T*w)

#def grad(w,X,y):
#    return 1.0/X.shape[0] * X.T * np.multiply( y, sigmoid(np.multiply(y, X*w)) - 1) + lamda*w

#def soft_threshod(w,mu):
#    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

### Gradient Descent
#w = np.matrix([0.0]*dim).T
#obj_GD = []
#max_iter = num_pass
#for t in range(0, max_iter):
#    obj_val = obj(w)
#    w = w - 2.0/(L+lamda) * grad(w, X, y)
    
#    obj_GD.append(obj_val.item())
    
#print('Objective function value is: {}'.format(obj_GD[-1]))

###Proximal gradient
#w = np.matrix([0.0]*dim).T
#obj_PGD = []
#max_iter = num_pass
#for t in range(0, max_iter):
#    obj_val = obj(w)
#    w = w - 2.0/(L+lamda) * grad(w, X, y)
#    w= soft_threshod(w,lamda/L)
    
#    obj_PGD.append(obj_val.item())
    
#print('Objective function value is: {}'.format(obj_PGD[-1]))

### Nesterovs' Accelerated Proximal Gradient
#w = np.matrix([0.0]*dim).T
#v = w
#obj_NA = []
#for t in range(0, max_iter):
#    obj_val = obj(w)
#    w_prev = w
#    w = v - 2.0/(L+lamda)*grad(w, X, y)
#    w = soft_threshod(w,lamda / L)
#    v = w + t/(t+3) * (w - w_prev)

#    obj_NA.append(obj_val.item())

#print('Objective function value is: {}'.format(obj_NA[-1]))

### Nesterovs' Accelerated Proximal Gradient with Backtracking
#w = np.matrix([0.0]*dim).T
#v = w
#obj_APG_LS = []
#L=1
#gamma = 1/L
#beta = 1.2
#for t in range(0, max_iter):
#    obj_val = obj(w)
#    w_prev = w
#    delta = 1
#    while (delta>1e-3):
#        gamma = 1/L
#        w = v - gamma * grad(v,X,y)    
#        w = soft_threshod(w,lamda * gamma)
#        delta = obj(w) - obj_val - grad(w_prev,X,y).T*(w-w_prev)- (L/2) * np.linalg.norm(w-w_prev)**2
#        L = L*beta
#    L = L/beta    
#    v = w + t/(t+3) * (w - w_prev)

#    obj_APG_LS.append(obj_val.item())
#    if (t%5==0):
#        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))

### ADMM
#w = np.matrix([0.0]*dim).T
#z = w
#u = w
#obj_ADMM = []
#rho = 5.0
#max_iter = num_pass
#for t in range(0, max_iter):
#    obj_val = obj(w)
#    w = np.linalg.inv((X.T)*X + rho*np.identity(dim))*(X.T*y + rho*z - u)
#    z= soft_threshod(w + u/rho,lamda/rho)
#    u = u + rho * (w-z)
#    obj_ADMM.append(obj_val.item())
#    if (t%5==0):
#        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))

#print('Objective function value is: {}'.format(obj_ADMM[-1]))

### Plot objective vs. iteration
#t = np.arange(0,num_pass)
#plt.plot(t, np.ones((len(t),1))*opt, 'k', linewidth = 2, label = 'Optimal')
#plt.plot(t, np.array(obj_GD), 'b', linewidth = 2, label = 'GD')
#plt.plot(t, np.array(obj_PGD), 'y', linewidth = 1, label = 'PGD')
#plt.legend(prop={'size':12})
#plt.xlabel('No. of Passes')
#plt.ylabel('Objective')
#plt.show()