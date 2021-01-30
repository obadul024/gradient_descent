import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA 


m , n = 2,6
A = np.random.rand(m,n)
x = np.random.rand(n,m)
print("X BEFORE",x)
b = np.random.rand(m,n)



def qfunc(A,x,b):
	try:
		prd1 = np.matmul(A,x)
		prd2 = 0.5 * (LA.norm(prd1) - b)
		return prd2
	except:
		print("Error QFunc ")


def dervF(A,x,b):
	try:
		prd1 = np.matmul(np.transpose(A),A);
		prd2 = np.matmul(np.transpose(prd1),x);
		#print(prd2.shape)
		prd3 = prd2 - np.matmul(np.transpose(A),b)
		return prd3
	except:
		print("Matrix Multiplication")



#stepsize = 0.01

def svgAlg(stepsize,epsilon,A,x,b):
	for i in range(10000):
		#print(LA.norm(dervF(A,x,b)))
		if (LA.norm(dervF(A,x,b)) > epsilon):
			x = x - stepsize*dervF(A,x,b);
	return x

epsilon = 0.001
stepsize = 0.01
Xmin = svgAlg(stepsize,epsilon,A,x,b)
print("Xmin NoW--------:",Xmin)
plt.plot(qfunc(A,Xmin,b),Xmin)
plt.ylabel('Minimized output Function dF ')

#print("Xmin Shape Now:",Xmin.shape,"\n")
#print("X shape before: ", x.shape,"\n")


