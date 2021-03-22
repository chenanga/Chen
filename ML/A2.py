#!/usr/bin/env python
# coding: utf-8

#

# **Question 1**: let $X$ be normally distributed. The parameters are mean $\mu$ and standard deviation $\sigma$, respectively. The likelihood function of $x$ is
# \begin{align}
# p(X = x;\mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
# \end{align}
# 
# There are $n$ independent identically distributed (i.i.d.) observation samples, which are $x_1, x_2, \cdots, x_n$. The likelihood function of them is
# 
# \begin{align}
# p(x_1, x_2,\cdots, x_n;\mu, \sigma) = \prod_{i=1}^np(X=x_i;\mu, \sigma) = \frac{1}{(2\pi)^{n/2}\sigma^n}e^{-\frac{\sum_{i=1}^n(x_i-\mu)^2}{2\sigma^2}}
# \end{align}
# 
# From the likelihood function and these observations, we are able to estimate parameter $\mu$ and $\sigma$. To make the estimate easy, we often use negative likelihood function, which is
# \begin{align}
# \label{eq1}
# L(\mu, \sigma) =-\text{ln}(p(x_1, x_2,\cdots, x_n;\mu, \sigma))= \frac{n}{2}\text{ln}(2\pi)+n\text{ln}(\sigma)+\frac{\sum_{i=1}^n(x_i-\mu)^2}{2\sigma^2}
# \end{align}
# 
# - Solve the following optimization problem
# \begin{align}
# \min_{\mu, \sigma} L(\mu, \sigma)
# \end{align}
# - We could also estimate parameters using gradient descent. Give the expressions of gradient of the negative log-likelihood with respect (w.r.t.) to $\mu$ and $\sigma$, respectively. Write a python method called "gradient" that gives the gradient vectors.
# - Gradient descent is an iterative algorithm. $\mu$ and $\sigma$ are updated as follows
# \begin{align}
# \mu_{i+1} &= \mu_i - \lambda \left.\frac{\partial L(\mu, \sigma)}{\partial \mu}\right|_{\mu=\mu_i, \sigma=\sigma_i}\\
# \sigma_{i+1} &= \sigma_i - \lambda \left.\frac{\partial L(\mu, \sigma)}{\partial \sigma}\right|_{\mu=\mu_i, \sigma=\sigma_i}
# \end{align}
# where $\lambda$ is the step length. Write a python method called "" to implement one step gradient descent update.
# - Write a python method called "gradient descent" to implement gradient descent to estimate the parameters.
# - Plot the norm of the gradient w.r.t $\mu$. The x axis is the number of steps and y axis represents the norm of the gradient w.r.t. $\mu$.
# - Plot the norm of the gradient w.r.t $\sigma$. The x axis is the number of steps and y axis represents the norm of the gradient w.r.t. $\sigma$.
# - Compare the parameter values estimated with the gradient descent algorithm with the values computed using closed form solution. Are your estimated parameters close to the closed form solutions? If not, something is likely wrong with your gradient descent algorithm.
# - Display the contour plot of the negative log-likelihood function along with the gradient descent trajectory. Examine the countour plot with descent path. Does the trajectory of the descent path always follow the maxium gradient direction? In other words, is the trajectory of the descent path always perpendicular to the countour lines of the log likelihood function?
# - Gradient descent uses all samples to calculate the gradient. Stochastic gradient descent (SDG) algorithm does not precisely compute gradient vector, but approximates the gradient vectors using partial samples. Like gradient descent algorithm, SDG is also an iterative method. In each iteration, $k$ samples are randomly selected to calculate gradient vector and parameters are updated via this gradient vector. Write a python method named "SDG" to implement SDG algorithm to estimate the parameters.
# - Plot the norm of the gradient w.r.t $\mu$. The x axis is the number of steps and y axis represents the norm of the gradient w.r.t. $\mu$.
# - Plot the norm of the gradient w.r.t $\sigma$. The x axis is the number of steps and y axis represents the norm of the gradient w.r.t. $\sigma$.
# - Compare the parameter values estimated with the gradient descent algorithm with the values computed using closed form solution. Are your estimated parameters close to the closed form solutions? If not, something is likely wrong with your gradient descent algorithm.
# - Display the contour plot of the negative log-likelihood function along with the gradient descent trajectory. Examine the countour plot with descent path. Does the trajectory of the descent path always follow the maxium gradient direction? In other words, is the trajectory of the descent path always perpendicular to the countour lines of the log likelihood function?

# In[53]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
import math


# Generating 1,000 normally distributed random variable with mean 4 and standard deviation 2. 

# In[54]:


np.random.seed(3)
x = np.random.normal(4, 2, 1000)


# negative_log_likelihood method compute the negative log-likelihood function value with given samples and parameters. 

# In[55]:


def negative_log_likelihood(x, mu, sigma):
    # input: x is a vector containing random numbers
    #        mu is the mean. mu is a scalar
    #        sigma is the standard deviation. sigma is a scalar
    # output: a scalar representing the negative log-likelihood function value defined as L(mu, sigma) above.
    # write your code below
    n,x=len(x),(x-mu)**2
    return n*math.log(2*math.pi)/2 +n*math.log(sigma) +np.sum(x)/(2*sigma**2)


# compute_gradient method gives a gradient vector containing the gradient of the negative log-likelihood function w.r.t. $\mu$ and $\sigma$, respectively.

# In[56]:


def compute_gradient(x, mu, sigma):
    # input: x is a vector containing random numbers
    #        mu is the mean. mu is a scalar
    #        sigma is the standard deviation. sigma is a scalar
    # output: a numpy array containing two elements, which are gradients w.r.t. mu and sigma, respectively.
    # write your code below
    return np.array([np.sum((x-mu)*(-2))/(2*sigma**2),len(x)/sigma - np.sum((x-mu)**2)/(sigma**3)])      


# gradient_descent is a method implementing gradient descent algorithm.

# In[57]:


def gradient_descent(x, mu, sigma, lambd):
    numIter = 1 # number of iterations
    M = [mu] # M stores updated value of mu
    S = [sigma] # S stores updated value of s2
    F = [negative_log_likelihood(x, mu, sigma)]
    gradNorm = []
    n = x.shape[0]
    while True:
        gradient = compute_gradient(x, mu, sigma) # compute gradient
        norming = np.linalg.norm(gradient) # compute norm of gradient
        if norming < 0.1 or numIter > 5000: # stop the while-loop is the norm␣→is under 0.1 or number of iterations is over 5000
            break
        else:
            gradNorm.append(norming)
            # to do: update mu and sigma
            # write your code below
            mu,sigma=mu-lambd*gradient[0],sigma-lambd*gradient[1]
            
            M.append(mu)
            S.append(sigma)
            F.append(negative_log_likelihood(x, mu, sigma))
            numIter += 1
    return numIter, gradNorm, M, S, F


# In[58]:


mu = 2
sigma = 1
lambd = 1.0e-5
numIter, gradNorm, M, S, F = gradient_descent(x, mu, sigma, lambd)


# The following python code plot the $\mu$, $\sigma$ and norm of the gradient vector along with number of iterations.

# In[59]:


def gradient_plot(numIter, gradNorm, M, S, F):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(221)    
    ax.plot(range(numIter), F)
    ax.set_title('Negative Log-likelihood v.s. Number of Iterations')
    ax.set_xlabel('Number of Iterations', fontsize = 12)
    ax.set_ylabel('Negative Log-Likelihood', fontsize = 12)
    ax = fig.add_subplot(222)    
    ax.plot(range(numIter-1), gradNorm)
    ax.set_title('Norm of Gradient v.s. Number of Iterations')
    ax.set_xlabel('Number of Iterations', fontsize = 12)
    ax.set_ylabel('Norm of Gradient', fontsize = 12)
    ax = fig.add_subplot(223) 
    ax.plot(range(numIter), M)
    ax.set_title('$\mu$ v.s. Number of Iterations')
    ax.set_xlabel('Number of Iterations', fontsize = 12)
    ax.set_ylabel('$\mu$', fontsize = 12)
    ax = fig.add_subplot(224) 
    ax.plot(range(numIter), S)
    ax.set_title('$\sigma$ v.s. Number of Iterations')
    ax.set_xlabel('Number of Iterations', fontsize = 12)
    ax.set_ylabel('$\sigma$', fontsize = 12)
    plt.show()


# In[60]:


gradient_plot(numIter, gradNorm, M, S, F)


# - Compare the parameter values estimated with the gradient descent algorithm with the values computed using closed form solution. Are your estimated parameters close to the closed form solutions? If not, something is likely wrong with your gradient descent algorithm.

# In[61]:


def contour_gradient(x, M, S)->None:
    mu = np.arange(0.5, 8.5, step = 0.05)
    sigma = np.arange(0.5, 4.5, step = 0.05)
    # Mu and Sigma are grids for location and scale
    Mu, Sigma = np.meshgrid(mu, sigma)
    # Compute Z, which is the log-likelihood value
    Z = np.zeros(Mu.shape)
    pdf = lambda x, y, z: norm.logpdf(x, loc = y, scale = z)
    for i, m in enumerate(mu):
        for j,s in enumerate(sigma):
            Z[j, i] = pdf(x, m, s).sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    cp = ax.contour(Mu, Sigma, Z, levels = 500, cmap = 'RdGy')
    ax.clabel(cp)
    ax.set_title('Contours Plot and Path')
    ax.set_xlabel('$\mu$', fontsize = 12)
    ax.set_ylabel('$\sigma$', fontsize = 12)
    ax.plot(M, S,'r.-')
    plt.show()


# In[62]:


contour_gradient(x, M, S)


# - Examine the countour plot with descent path. Does the trajectory of the descent path always follow the maxium gradient direction? In other words, is the trajectory of the descent path always perpendicular to the countour lines of the log likelihood function?

# In[63]:


def SDG(x, mu, sigma, lambd, k):
    numIter = 1 # number of iterations
    M = [mu] # M stores updated value of mu
    S = [sigma] # S stores updated value of s2
    F = [negative_log_likelihood(x, mu, sigma)]
    gradNorm = []
    n = x.shape[0]
    while True:
        # randomly select k samples from all samples store in x. 
        #The selected samples form a vector called samples. 
        samples = np.random.choice(x,k)
        
        gradient = compute_gradient(samples, mu, sigma) # compute gradient
        gradient = gradient/k*n
        norming = np.linalg.norm(gradient) # compute norm of gradient
        if norming < 0.1 or numIter > 5000: # stop the while-loop is the norm is under 0.1 or number of iterations is over 5000
            break
        else:
            gradNorm.append(norming)
            # update mu and sigma. Write your code below
            mu,sigma=mu-lambd*gradient[0],sigma-lambd*gradient[1]
            
            M.append(mu)
            S.append(sigma)
            F.append(negative_log_likelihood(x, mu, sigma))
            numIter += 1
    return numIter, gradNorm, M, S, F


# In[64]:


mu = 2
sigma = 1
lambd = 1.0e-5
k = 100
numIter, gradNorm, M, S, F = SDG(x, mu, sigma, lambd, k)


# In[65]:


gradient_plot(numIter, gradNorm, M, S, F)


# - Compare the parameter values estimated with the gradient descent algorithm with the values computed using closed form solution. Are your estimated parameters close to the closed form solutions? If not, something is likely wrong with your gradient descent algorithm.

# In[66]:


contour_gradient(x, M, S)


# - Examine the countour plot with descent path. Does the trajectory of the descent path always follow the maxium gradient direction? In other words, is the trajectory of the descent path always perpendicular to the countour lines of the log likelihood function?

# Yes, the trajectory of the descent path always perpendicular to the countour lines of the log likelihood function.
