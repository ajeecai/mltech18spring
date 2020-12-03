#!/usr/bin/env python3
import numpy as np
#need "pip3 install OSQP" to use osqp for semi positive definite process
from qpsolvers import solve_qp, sparse_solvers,available_solvers

# Xn, Xm is vector of data dot n,m
# return the kernel result of  (1 + 2x^T x')^2

def k_xn_xm(xn, xm):
    #print(xn,xm)
    zz = (1+2*xn.dot(xm)) **2
    #print("k is ",z)
    return zz

# Be attention, all data should be in double type
label = [-1., -1., -1., 1., 1., 1., 1.]
training_data = np.array([[1.,0.],[0.,1.],[0.,-1.],[-1.,0.],[0.,2.],[0.,-2.],[-2.,0.]])

Q = []
for n, x_n in enumerate(training_data):
    qn = []
    for m, x_m in enumerate(training_data):
        qnm = label[n]*label[m]*k_xn_xm(x_n, x_m)
        qn.append(qnm)
    #print(qn)
    Q.append(qn)

Q = np.array(Q)
#print(Q)
# vector doesn't need to transpose? 
# make sure the dimensions of all params should match requirement
p = np.array([-1.] * len(label))
#print(p)
b = np.array([0.])
A = np.array(label)
#print(A)
h = np.array([0.] * len(label))
G = np.identity(len(label)) * -1.
#print(G)

print(sparse_solvers,available_solvers)

# choose osqp solver
alphas = solve_qp(Q, p, G, h, A, b, solver='osqp')
print("QP solution:", alphas)

def normalize_alpha(a,b):
    if np.abs(a) > b:
        return a
    else:
        return 0

na = np.vectorize(normalize_alpha,otypes=[float])
alphas = list(na(alphas,1e-3))
print("QP solution after normalized:", alphas)