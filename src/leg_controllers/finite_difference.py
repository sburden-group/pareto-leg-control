import numpy as np

def central_difference(f, x, h):
    n = len(x)
    f_x = f(x)
    f_shape = np.shape(f_x)
    fdiff = np.zeros((f_shape)+(n,))
    delta = np.eye(n)
    for i in range(n):
        fdiff[...,i] = (-f(x-h*delta[:,i])+f(x+h*delta[:,i]))/(2*h)
    return fdiff

def hessian(f,x,h):
    n = len(x)
    f_x = f(x)
    f_shape = np.shape(f_x)
    hess = np.zeros((f_shape)+(n,n))
    delta = np.eye(n)
    for i in range(n):
        for j in range(n):
            dxi = h*delta[:,i]
            dxj = h*delta[:,j]
            hess[...,i,j] = (f(x+dxi+dxj)-f(x+dxi)-f(x+dxj)+f_x)/h**2
    
    return (hess+np.transpose(hess,(0,2,1)))/2
