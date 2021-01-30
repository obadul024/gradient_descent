#!/usr/bin/env python
# coding: utf-8

# #                                    **Deeplearning Assingment # 2 Gradient Descent**

# **Gradient Descent Method Implementation using Numpy,
# by Engr. Obaidullah, CS 1947**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA 


# In[2]:


# Linear Equation Definition as per Eq.4.21

def qfunc(A,x,b):
    try:
        prd1 = np.matmul(A,x)
        #print(prd1.shape, b.shape)
        prd2 = LA.norm(np.subtract(prd1,b))**2
        #print(prd2)
        return prd2*0.5
    except:
        print("Error QFunc ")
        


# In[3]:


# Derivative Function from Eq. 4.22 in the book 



def dervF(A,x,b):
    try:
        prd1 = np.matmul(A,x)
        #print(A.shape, x.shape)
        #print("\n",prd1.shape)
        
        prd2 = np.matmul(np.transpose(A),prd1)
        #print("\nPRD2",prd2.shape)
        prd3 = np.matmul(np.transpose(A),b)
        #print("\n",prd3.shape)
        #print("Matrix shapes::At*A ## AtA*x ## Atb",prd1.shape,prd2.shape,prd3.shape)
        prd4 = np.subtract(prd2,prd3)
        
        return prd4
    except:
        print("Matrix Multiplication Error")


# In[34]:


# DONOT RUN THIS CELL
# SGD Using For Loop

def sgdAlg(stepsize,epsilon,A,x,b):
    for i in range(10000):
        #print(LA.norm(dervF(A,x,b)))
        if (LA.norm(dervF(A,x,b)) > epsilon):
            x = x - stepsize*dervF(A,x,b);
    return x


# In[35]:


# SGD Using While

def sgdAlg(stepsize,epsilon,A,x,b):
    while(LA.norm(dervF(A,x,b)) > epsilon):
        x = x - stepsize*dervF(A,x,b);
    return x


# In[43]:


# Constant Variables setup And Matrix Size definitions

m , n = 1,20

epsilon = 0.0001
stepsize = 0.01
A = np.random.rand(m,n)
x = np.random.rand(n)
b = np.random.rand(1,m)
#derivative = dervF(A,x,np.transpose(b))


# In[44]:


#minimized values of X from the SGD, stochastic Gradient Descend Method

Xmin = sgdAlg(stepsize,epsilon,A,x,np.transpose(b)) 


# In[48]:


# Passing Minimized Xmin through Q Function -> Ax - b = 0
linear_eq_out = np.array([qfunc(A,i,np.transpose(b)) for i in Xmin])
print(linear_eq_out)


# In[46]:





# In[47]:



xaxis = Xmin
plt.plot(linear_eq_out)
plt.ylabel('Output of Function ')


# In[ ]:


# Below code is for testing purposes only Donot Run


# #print(A)
# #print("\n",b)
# 
# print("Derivative: ",derivative.shape)
# print(qoutput.shape)
# #print("Xmin NoW--------:",Xmin.shape)
# #qoutput = qfunc(A,Xmin[0],np.transpose(b))
# epsilon = 0.001
# stepsize = 0.01
# m , n = 2,4
# A = np.random.rand(m,n)
# x = np.random.rand(n)
# b = np.random.rand(m)
# 
# #print("X BEFORE",x.shape)
# x1 = np.arange(9.0).reshape((3, 3))
# x2 = np.arange(3.0).reshape(3,1)
# print(x1,x2,"\n")
# print(np.subtract(x1, x2))
# 
