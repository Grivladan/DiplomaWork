import numpy as np
import math

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


# Nesterovs' Accelerated Proximal Gradient
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

# ADMM
def ADMM(A, y, f_grad, prox):
    w = np.matrix([0.0]*dim).T
    z = w
    u = w
    obj_ADMM = []
    rho = 5.0
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = np.linalg.inv((X.T)*X + rho*np.identity(dim))*(X.T*y + rho*z - u )
        z= prox(w + u/rho,lamda/rho)
        u = u + rho * (w-z)
        obj_ADMM.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_ADMM

## Gradient Descent
def gradient_descent(X, y, grad, step):
    w = np.matrix([0.0]*dim).T
    obj_GD = []
    for t in range(0, max_iter):
        obj_val = obj(w)
        w = w - step * grad(X, y, w)
        obj_GD.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_GD

### Nesterovs' Accelerated Proximal Gradient with Backtracking
def backtracking_nesterov_acceleration(X, y, grad, prox):
    w = np.matrix([0.0]*dim).T
    v = w
    obj_APG_LS = []
    L = 1
    beta = 1.2
    for t in range(0, max_iter):
        obj_val = obj(w)
        w_prev = w
        delta = 1
        while (delta>1e-3):
            gamma = 1/L
            w = v - gamma * grad(X,y,v)    
            w = prox(w,lamda * gamma)
            delta = obj(w) - obj_val - grad(X, y, w_prev).T*(w-w_prev)- (L/2) * np.linalg.norm(w-w_prev)**2
            L = L*beta
        L = L/beta    
        v = w + t/(t+3) * (w - w_prev)

        obj_APG_LS.append(obj_val.item())
        if (t%5==0):
            print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))
    return obj_APG_LS 
