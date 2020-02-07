import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Example 3

x = [4.6,6.0,2.0,5.8]
sigma = [2.0,1.5,5.0,1.0]	
mu = np.linspace(0,10,1000)
mu_min = 0.
mu_max = 10.

def prior(mu_min,mu_max,mu):
	A = 1/(mu_max-mu_min)
	prior_array = np.zeros(len(mu))
	ind_A = np.where((mu <= mu_max)&(mu >= mu_min))
	prior_array[ind_A] = A

	return prior_array

def likeli(x,sigma,mu):
	like = np.ones(len(mu))
	for i in range(0,len(x)):
		mult = np.exp(-((x[i]-mu)**2)/(2.*(sigma[i]**2)))/np.sqrt(2*np.pi*(sigma[i]**2))
		like *= mult

	return like

def posterior(x,sigma,mu,mu_min,mu_max):
	post_pre = likeli(x,sigma,mu)*prior(mu_min,mu_max,mu)
	norm = integrate.trapz(post_pre,mu)

	return post_pre/norm

def max_log(x,y):
	delta_x = x[1]-x[0]
	index = np.argmax(y)
	d = (y[index+1]-2*y[index]+y[index-1])/(delta_x**2)

	return x[index], 1.0/np.sqrt(-d)

plt.figure(figsize=(10,10))
prob = posterior(x,sigma,mu,mu_min,mu_max)
est,error = max_log(mu,np.log(prob))
plt.plot(mu,prob)
plt.ylabel(r'$P(\mu|\{x_k\},\{\sigma_k\},I)$')
plt.xlabel(r'$\mu$')
plt.title(r'$\mu = ({:.2f} \pm {:.2f})$'.format(est,error))
plt.grid(1)

plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
plt.savefig('mean.png', hhbox = 'tight')