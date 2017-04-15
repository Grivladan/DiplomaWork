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
step = 1.0 / L

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
    hessian = np.ones((dim,dim+1))
    for i in range(dim):
        hessian[i,0] = 1
    for i in range(dim):
        for j in range(dim):
            hessian[i, j+1] = X[i, j]
    return np.dot(hessian,hessian.T)

w = cvx.Variable(dim)
loss = cvx.sum_squares(X*w-y)/2 + lamda * cvx.norm(w,1)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))

#Proximal gradient
def proximal_grad(A, y, f_grad, prox, step):
    w = np.matrix([0.0]*dim).T
    obj_PG = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - step* f_grad(A, y, w)
        w= prox(w,lamda/L)
    
        obj_PG.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_PG

#forward-backward envelope
def forward_backward(A, y, f_grad, prox, step):
    w = np.matrix([0.0]*dim).T
    rho = 0.7
    obj_FBE = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        z = w - step* f_grad(A, y, w)
        z= prox(z,lamda/L)
        w = z + rho*(z - w)
    
        obj_FBE.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_FBE

#Nesterow acceleration forward-backward envelope
def accelerated_forward_backward(A, y, f_grad, prox, step):
    w = np.matrix([0.0]*dim).T
    obj_AFBE = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        z = w - step* f_grad(A, y, w)
        z= prox(z,lamda/L)
        alpha = t / (t+3.0)
        w = z + alpha*(z - w)
    
        obj_AFBE.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_AFBE

pA = np.linalg.pinv(X) # pseudo-inverse. Equivalent to pA = A.T.dot(inv(A.dot(A.T)))
def prox_f (A, x, y) :
    return x + pA.dot(y-A.dot(x))
#Douglas_Rachford envelope
def douglas_rachford(A, y, f_grad, f_prox, g_prox, step):
    w = np.matrix([0.0]*dim).T
    s = np.matrix([0.0]*dim).T
    rho = 1
    obj_ = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = f_prox(s, y)
        s = s + rho * (g_prox(2*w - s, gamma) - w)

        obj_DRE.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_DRE

#Proximal Newton
def proximal_newton(A, y, f_grad, f_hess, prox, step):
    w = np.matrix([0.0]*dim).T
    obj_PN = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - step * np.linalg.inv(f_hess(A))*f_grad(A, y, w)
        w= prox(w,lamda/L)
    
        obj_PN.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_PN

#ISTA
def iterative_shrinkage(A, y, f_grad, prox, step):
    w = np.matrix([0.0]*dim).T
    obj_ISTA = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - step * A.T*(A*w - y)
        w= prox(w,lamda / L)
    
        obj_ISTA.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_ISTA


#Nesterovs' Accelerated Proximal Gradient
def accelerated_proximal_gradient(A, y, f_grad, prox, step):
    w = np.matrix([0.0]*dim).T
    v = w
    obj_APG = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w_prev = w
        w = v - step * f_grad(A,y,v)
        w = soft_threshod(w,lamda * step)
        v = w + t/(t+3.0) * (w - w_prev)

        obj_APG.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_APG

 #ADMM
def ADMM(A, y, f_grad, prox):
    w = np.matrix([0.0]*dim).T
    z = w
    u = w
    obj_ADMM = []
    rho = 5.0
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
obj_PG = proximal_grad(X, y, f_grad, soft_threshod,step)

print('Accelerated proximal gradient')
obj_APG = accelerated_proximal_gradient(X, y, f_grad, soft_threshod, step)

"""
print('Proximal Newton')
obj_APG = proximal_newton(X, y, f_grad, hess, soft_threshod, step)
"""

print('Ista algorithm')
obj_ISTA = iterative_shrinkage(X, y, f_grad, soft_threshod, step)
 
print('Alternating direction multipliers method')
obj_ADMM = ADMM(X, y, f_grad, soft_threshod)

print('Forward-Backward envelope')
obj_FBE = forward_backward(X, y, f_grad, soft_threshod, step)

print('Accelerated forward-backward envelope')
obj_AFBE = accelerated_forward_backward(X, y, f_grad, soft_threshod, step)

t = np.arange(0, max_iter)
fig, ax = plt.subplots(figsize = (9, 6))
plt.semilogy(t, np.array(obj_PG) - opt, 'b', linewidth = 2, label = 'Proximal Gradient')
plt.semilogy(t, np.array(obj_APG) - opt, 'c--', linewidth = 2, label = 'Accelerated Proximal Gradient')
plt.semilogy(t, np.array(obj_ISTA) - opt, 'g', linewidth = 2, label = 'ISTA')
plt.semilogy(t, np.array(obj_ADMM) - opt, 'r', linewidth = 2, label = 'ADMM')
plt.semilogy(t, np.array(obj_FBE) - opt, 'r', linewidth = 2, label = 'Forward-Backward envelope')
plt.semilogy(t, np.array(obj_AFBE) - opt, 'g', linewidth = 2, label = 'Accelerated forward-backward')
plt.legend(prop={'size':12})
plt.xlabel('Iteration')
plt.ylabel('Objective error')
plt.show()

#logistic loss 
#from sklearn import datasets
#from sklearn.preprocessing import normalize
#from scipy.special import expit as sigmoid

#N = 10000
#dim = 50
#lamda = 1e-4
#np.random.seed(20)
#w = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim))).T
#X = np.matrix(np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N))
#X = np.matrix(normalize(X, axis=1, norm='l2'))
#y = 2 * (np.random.uniform(size = (N, 1)) < sigmoid(X*w)) - 1

#w = cvx.Variable(dim)
#loss = 1.0 / N * cvx.sum_entries(cvx.logistic(-cvx.mul_elemwise(y, X*w))) + lamda * cvx.norm(w,2)

#problem = cvx.Problem(cvx.Minimize(loss))
#problem.solve(verbose=True, abstol=1e-15) 
#opt = problem.value
#print('Optimal Objective function value is: {}'.format(opt))

#L = lamda + 1.0/4;
#max_iter = 100
#step = 2.0/(L+lamda)

### Define the objective and gradient oracles. 
#def obj(w):
#    return 1.0/N * np.sum( np.log(1 + np.exp(-np.multiply(y, (X*w)))) ) + 1.0/2 * lamda * (w.T*w)

#def grad(X,y,w):
#    return 1.0/X.shape[0] * X.T * np.multiply( y, sigmoid(np.multiply(y, X*w)) - 1) + lamda*w

#def soft_threshod(w,mu):
#    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

## Gradient Descent
#def gradient_descent(X, y, grad, step):
#    w = np.matrix([0.0]*dim).T
#    obj_GD = []
#    for t in range(0, max_iter):
#        obj_val = obj(w)
#        w = w - step * grad(X, y, w)
#        obj_GD.append(obj_val.item())
#        if (t%5==0):
#            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
#    return obj_GD

#LL = np.diag(X.T*X)/4+lamda
### Cyclic Coordinate Gradient Descent
#def cyclic_gradient(X, y, grad, step):
#    w = np.matrix([0.0]*dim).T
#    obj_CCGD = []
#    for t in range(0, max_iter):
#        obj_val = obj(w)
#        for i in range(0,dim):
#            w[i] = w[i] - step * grad(X, y, w)[i]
#            # larger stepsize
#            w[i] = w[i] - 1/LL[i] * grad(X,y,w)[i]

#        obj_CCGD.append(obj_val.item())
#        if (t%5==0):
#            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
#    return obj_CCGD

#### Nesterovs' Accelerated Proximal Gradient with Backtracking
#def backtracking_nesterov_acceleration(X, y, grad, prox):
#    w = np.matrix([0.0]*dim).T
#    v = w
#    obj_APG_LS = []
#    L=1
#    gamma = 1/L
#    beta = 1.2
#    for t in range(0, max_iter):
#        obj_val = obj(w)
#        w_prev = w
#        delta = 1
#        while (delta>1e-3):
#            gamma = 1/L
#            w = v - gamma * grad(X,y,v)    
#            w = prox(w,lamda * gamma)
#            delta = obj(w) - obj_val - grad(X, y, w_prev).T*(w-w_prev)- (L/2) * np.linalg.norm(w-w_prev)**2
#            L = L*beta
#        L = L/beta    
#        v = w + t/(t+3) * (w - w_prev)

#        obj_APG_LS.append(obj_val.item())
#        if (t%5==0):
#            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
#    return obj_APG_LS 

#print('Gradient descent')
#obj_GD = gradient_descent(X, y, grad, step)

#print('Proximal gradient')
#obj_PG = proximal_grad(X, y, grad, soft_threshod, step)

#print('Accelerated proximal gradient')
#obj_APG = accelerated_proximal_gradient(X, y, grad, soft_threshod, step)

#print('ADMM method')
#obj_ADMM = ADMM(X, y, grad, soft_threshod)

#print('Nesterov acceleration with backtracking')
#obj_APG_LS = backtracking_nesterov_acceleration(X, y, grad, soft_threshod)

#print('Cyclic Coordinate gradient')
#obj_CCGD = cyclic_gradient(X, y, grad, step)

##print('ISTA method')
##obj_ISTA = iterative_shrinkage(X, y, grad, soft_threshod, step)

### Plot objective vs. iteration
#t = np.arange(0,max_iter)
#plt.plot(t, np.ones((len(t),1))*opt, 'k', linewidth = 2, label = 'Optimal')
#plt.plot(t, np.array(obj_GD), 'b', linewidth = 2, label = 'GD')
#plt.plot(t, np.array(obj_PG), 'y', linewidth = 1, label = 'PGD')
#plt.plot(t, np.array(obj_APG), 'r', linewidth = 2, label = 'APG')
#plt.plot(t, np.array(obj_APG_LS), 'r--', linewidth = 2, label = 'APG_LS')
#plt.plot(t, np.array(obj_ADMM), 'g', linewidth = 2, label = 'ADMM')
#plt.plot(t, np.array(obj_CCGD), 'c', linewidth = 2, label = 'Cyclic Coordinate Gradient')
#plt.legend(prop={'size':12})
#plt.xlabel('No. of Passes')
#plt.ylabel('Objective')
#plt.show()

