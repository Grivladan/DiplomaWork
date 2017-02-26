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

w = np.matrix([0.0]*dim).T
obj_PG = []
for t in range(0, max_iter):
    obj_val = obj(w)
    w = w - (1/L)* f_grad(w)
    w= soft_threshod(w,lamda/L)
    
    obj_PG.append(obj_val.item())
    if (t%5==0):
        print('iter= {},\tobjective= {:3f}'.format(t, obj_val.item()))