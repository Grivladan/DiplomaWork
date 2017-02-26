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

t = np.arange(0, max_iter)
fig, ax = plt.subplots(figsize = (9, 6))
plt.semilogy(t, np.array(obj_PG), 'b', linewidth = 2, label = 'Proximal Gradient')
plt.semilogy(t, np.array(obj_APG), 'c--', linewidth = 2, label = 'Accelerated Proximal Gradient')
plt.legend(prop={'size':12})
plt.xlabel('Iteration')
plt.ylabel('Objective error')
plt.show()