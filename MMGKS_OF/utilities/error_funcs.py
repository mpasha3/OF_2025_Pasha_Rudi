import numpy as np
from scipy import linalg as la
def vec2(img):
    if len(img.shape)==2:
        vector = img.reshape(img.shape[0]*img.shape[1])
    else:
        vector=img
    return vector
def mu(x):
    return np.mean(vec2(x))
def sigma(x):
    return np.std(vec2(x),ddof=1)
def ssigma(x,y):
    x= vec2(x)
    y=vec2(y)
    N=len(x)
    return (1/(N-1))*np.sum((x-mu(x))*(y-mu(y)))
def l(x,y):
    return 2*mu(x)*mu(y)/(mu(x)**2  + mu(y)**2)  
def c(x,y):
    return 2*sigma(x)*sigma(y)/(sigma(x)**2  + sigma(y)**2)  
def s(x,y):
    return ssigma(x,y)/(sigma(x)*sigma(y))
def ssim(x,y,alpha=1,beta=1,gamma=1):
    return l(x,y)**alpha * c(x,y)**beta * s(x,y)**gamma
def ssim_all(X,Y,nt,nx,ny,alpha=1,beta=1,gamma=1):
    X = np.array(X).reshape(nt,nx,ny)
    Y = np.array(Y).reshape(nt,nx,ny)
    return np.mean([ssim(X[i],Y[i],alpha,beta,gamma) for i in range (nt)])
def rre(x,x_true):
    return la.norm(x-x_true)/la.norm(x_true)
def rre_all(X,X_true,nt,nx,ny):
    X = np.array(X).reshape(nt,nx,ny)
    X_true = np.array(X_true).reshape(nt,nx,ny)
    return np.mean([rre(X[i],X_true[i]) for i in range (nt)])