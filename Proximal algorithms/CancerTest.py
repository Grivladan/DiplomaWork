import numpy as np
import cvxpy as cvx
import operator 

np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3, linewidth = 120)

F = open("ALLDATA.DAT", "r" )
testData = F.readlines()

X = np.zeros((36,16))
y = np.zeros(36)
healthX = np.zeros((10,16))

for i in range(4,40):
    tmpArr = testData[i].split()
    y[i-4]=float(tmpArr[1])
    for j in range (2,18):
        X[i-4][j-2] = float(tmpArr[j])

corrX = np.corrcoef(X.T)
print(corrX)

X = np.delete(X, 15, 1)

#L = (np.linalg.svd(X)[1][0])**2
#max_iter = 100
#step = 1.0 / L
#lamda = 1.0/10;

#def obj(w):
#    r = X*w-y;
#    return np.sum(np.multiply(r,r))/2 +  lamda * np.sum(np.abs(w))

#def f_grad(X, y, w):
#    return  X.T*(X*w-y) 

#def soft_threshod(w,mu):
#    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

#w = cvx.Variable(15)
#loss = cvx.sum_squares(X*w-y)/2 + lamda * cvx.norm(w,1)

#problem = cvx.Problem(cvx.Minimize(loss))
#problem.solve(verbose=True) 
#opt = problem.value
#print('Optimal Objective function value is: {}'.format(opt))

#tmpX = X

#for i in range(0,14):
#     X = np.delete(tmpX, i, 1)
#     w = cvx.Variable(14)
#     loss = cvx.sum_squares(X*w-y)/2 + lamda * cvx.norm(w,1)
#     problem = cvx.Problem(cvx.Minimize(loss))
#     problem.solve(verbose=True) 
#     opt = problem.value
#     print('Optimal Objective function value is: {}'.format(opt))

#for i in range(44, 54):
#    print(testData[i])

#solve with cvxpy library
w = cvx.Variable(15)
w0 = cvx.Variable(1)
loss = cvx.sum_squares(y - X*w - w0)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))


#solve with least squares method
X = np.c_[ np.ones(36), X ] 
least_squares_solution = np.linalg.lstsq(X, y) 
least_squares_coef = least_squares_solution[0]
least_squares_result = least_squares_solution[1]
print(least_squares_coef)
print(least_squares_result)

tmpX = X

characteristic_values = {}
for i in range(1,15):
     X = np.delete(tmpX, i, 1)
     characteristic_values[i] = np.linalg.lstsq(X, y)[1]

sorted_characteristic_values = sorted(characteristic_values.items(), key=operator.itemgetter(1))
for item in enumerate(sorted_characteristic_values):
    print('optimal value without characteristic x{} is {}'.format(item[1][0], item[1][1]))
