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

k_xn_xm(training_data[0], training_data[1])

Q = []
for n, phi_n in enumerate(training_data):
    qn = []
    for m, phi_m in enumerate(training_data):
        qnm = label[n]*label[m]*k_xn_xm(phi_n, phi_m)
        # qnm = k_xn_xm(phi_n, phi_m)
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

 # choose osqp, make sure it is in slot 0. Could qpsolver use dictionary instead of list for this?
print("QP solution:", solve_qp(Q, p, G, h, A, b, solver=sparse_solvers[0]))


