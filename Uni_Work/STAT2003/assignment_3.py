"""
# STAT2003 Assignment 3
# Q1
import numpy as np
import matplotlib.pyplot as plt

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

# Q5
import numpy as np

count = 100_000
m, q, mu = 100, 1/5, 30

n_samples = np.random.binomial(m,q,size=count)

s_samples = np.zeros(count)

for i in range(count):
    n = n_samples[i]
    if n > 0:
        s_samples[i] = np.sum(np.random.exponential(mu,size=n))
    else:
        s_samples[i] = 0


print(f"Theoretical E[S] = {600}")
print(f"Simulated E[S] = {s_samples.mean():.2f}")
print(f"Theoretical Var(S) = {32400}")
print(f"Simulated Var(S) = {s_samples.var():.2f}")
"""

import numpy as np
from scipy.stats import norm

print(f"{'x':>4} {'Exact P(Z>x)':>15} {'Chernoff bound':>15} {'Ratio':>10}")
print("_"*50)

for x in [1,2,3,4,5]:
    exact_prob = norm.sf(x)
    bound = np.exp(-x**2 / 2)
    ratio = bound / exact_prob
    print(f"{x:>4} {exact_prob:>15.3e} {bound:>15.3e} {ratio:>10.3f}")