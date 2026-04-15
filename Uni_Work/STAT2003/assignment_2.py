"""
# Q1 D)
import numpy as np
N = 10
trials = 100000
x = np.arange(1, N+1)
probabilities = x / x.sum()
samples = np.random.choice(x, trials, p=probabilities)

print(f"Sample Mean: {samples.mean():.3f}")
print(f"Sample Variance: {samples.var():.3f}")

mean_est = (2*N+1)/3
var_est = ((N-1)*(N+2))/18

print(f"Theoretical Mean: {mean_est:.3f}")
print(f"Theoretical Variance: {var_est:.3f}")


# Q1 E)
import numpy as np
import matplotlib.pyplot as plt

trials = 100000
# i) Draw G:
G = np.random.geometric(p=1/3, size=trials)
# ii) G = g, draw Y from g with probs proportional to position:
array = np.zeros(trials, dtype=int)
for i in range(trials):
    # G = g
    g = G[i]
    y_init = 10*(g-1)+1
    # 10 positions in each block
    # (y-10*(g-1))/55
    positions = np.arange(1,11)
    probabilities = positions/55
    local = np.random.choice(positions, p=probabilities)

    array[i] = y_init + local - 1

print(f"Estimated Mean (E[Y]): {array.mean():.3f}")

plt.hist(array[array <= 50], bins = np.arange(1, 52)-0.5, density=True, color='steelblue', edgecolor='white')
plt.xlabel('y')
plt.ylabel('Probability')
plt.title('Histogram of Y (support of {1,2,...,50})')
plt.show()


# Q2 D)
from scipy.stats import binom

D = 0.002
Dc = 1 - D
n = 5
sensitivity = 0.98
false_positive = 0.12

for k in range(6):
    y_given_D = binom.pmf(k, n, sensitivity)
    y_given_Dc = binom.pmf(k, n, false_positive)
    d_given_y = (y_given_D * D)/(y_given_D * D + y_given_Dc * Dc)
    print(f" k = {k}, P(D|Y) = {d_given_y:.6f}")


# Q3 e)
import numpy as np

trans_matrix = np.array([[0.5,0.3,0.2],
                         [0.1,0.6,0.3],
                         [0.4,0.2,0.4]])
steps = 100000
E = np.array([1,2,3])

# Create empty matrix and start state.
totals = np.zeros(3)
x = 0

for i in range(steps):
    x = np.random.choice([0,1,2], p=trans_matrix[x])
    totals[x] += 1

proportion = totals/steps

print(f"Stationary Distribution Simulation after {steps} steps:")
for i in range(len(E)):
    print(f"  State {i+1}: {proportion[i]:.3f}")


# Q4 c)
import numpy as np

initial_state = 2
p = 1/3
q = 1 - p
N = 4
trials = 100000
successes = 0

for i in range(trials):
    fortune = initial_state
    while 0 < fortune and fortune < N:
        if np.random.random() < p:
            fortune += 1
        else:
            fortune -= 1

    if fortune == N:
        successes += 1

probability = successes/trials

print(f"The gambler succeeded in reaching their goal fortune with a probability of P(reach N | X_0 = 2) = {probability}.")
"""

# Q5 d)
from scipy.optimize import brentq

def CDF(x):
    return x**3 * (4-3*x)

low, high = 0, 1
for i in range(100):
    middle = (low + high)/2
    if CDF(middle) < 0.5:
        low = middle
    else:
        high = middle

print(f"The median is {middle:.3f}.")
