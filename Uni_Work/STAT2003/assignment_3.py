# STAT2003 Assignment 3
# Q1
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10**5
alpha = 2

Z = np.random.exponential(scale=1/alpha, size=n)
X = np.exp(Z)

x_vals = np.linspace(1, 50, 1000)

emp_suv = np.array([np.mean(X > x) for x in x_vals])

theo_suv = (x_vals)**(-alpha)

plt.figure(figsize=(10, 6))
plt.loglog(x_vals, emp_suv, label='Empirical')
plt.loglog(x_vals, theo_suv, 'r--', label='Theoretical')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.title('Empirical vs Theoretical Survival Function')
plt.show()

