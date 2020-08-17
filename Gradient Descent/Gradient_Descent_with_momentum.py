import numpy as np
import time

X=[0.5,2.5]
Y=[0.2,0.9]

def f(w,b,x):
    return 1.0/(1.0+np.exp(-(w*x+b)))

def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err+=0.5*(fx-y)**2
    return err

def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*(fx)*(1-fx)

def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*(fx)*(1-fx)*x

def do_gradient_descent():
    
    tic = time.time()
    w,b,eta,max_epoch = -7,-1,0.1,1000
    
    for i in range(max_epoch):
        dw=0
        db=0
        for x,y in zip(X,Y):
            dw+=grad_w(w, b, x, y)
            db+=grad_b(w, b, x, y)
            
        w = w - eta*dw
        b = b - eta*db
    toc = time.time()
    
    print("Cost for vanila gradient descent : ",error(w, b))
    print("Time required in seconds : ",toc - tic )
    
def do_momentum_gradient_descent():
    
    tic = time.time()
    w,b,eta,max_epoch = -7,-1,0.1,1000
    gama= 0.1
    w_pre, b_pre = 0, 0
    
    for i in range(max_epoch):
        dw=0
        db=0
        
        for x,y in zip(X,Y):
            dw+=grad_w(w, b, x, y)
            db+=grad_b(w, b, x, y)
            
        w = ((gama*w_pre) + (eta*dw))
        b = ((gama*b_pre) + (eta*db))
        
        w_pre=w
        b_pre=b
    
    toc = time.time()
    
    print("Cost for momentum gradient descent : ",error(w, b))
    print("Time required in seconds : ",toc - tic )
    
    
do_gradient_descent()
print("======================================")
do_momentum_gradient_descent()