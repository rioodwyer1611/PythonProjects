"""
# Question 2 c)
import random
def trial():
    x1 = random.randint(0,8)
    x2 = random.randint(0,8)
    x3 = random.randint(0,8)
    if x1 + x2 + x3 == 8:
        return 1
    else:
        return 0

N = 10**6
print(f"The result is {sum([trial() for _ in range(N)])/N:.3f}")


# Question 4 d)
from math import comb

def probability_of_b(n, r):
    total = 0
    # run loop
    for k in range(n + 1):
        total += ((-1)**k) * comb(n, k) * ((n - k) / n)**r
    return total

n = 25
r = 75

print(f"The outcome of n = {n} and r = {r} is {probability_of_b(n,r):.3f}")

# Question 4 e)
from math import comb

n = 25

def probability_of_b(n, r):
    total = 0
    # run loop
    for k in range(n + 1):
        total += ((-1)**k) * comb(n, k) * ((n - k) / n)**r
    return total

def find_lowest_r(n):
    r = 0
    while probability_of_b(n, r) <= 0.95:
        r += 1
    return r

lowest_r = find_lowest_r(n)

print(f"{lowest_r} is the minimal value of r. The value of P(B) at this r is {probability_of_b(n, lowest_r):.6f}")
print(f"To check, the value previous is {probability_of_b(n, lowest_r-1):.6f}")

# Question 5 c)
from math import comb

def pmf(x):
    return comb(20,x) * (1/4)**x * (3/4)**(20-x)

probability = 0
probability2 = 0

for x in range (10, 21):
    probability += pmf(x)

print(f"The probability that the student passes is {probability:.3f}")


# Question 6 e)
from scipy.stats import bernoulli
import numpy as np

def geom(p):
    count = 0
    while True:
        count += 1
        if bernoulli.rvs(p) == 1:
            return count
        
N = 10**6
p = 1/6

values = [geom(p) for _ in range(N)]
mean = np.mean(values)
variance = np.var(values)

print(f"Sample Mean: {mean:.3f} Sample Variance: {variance:.3f}")
print(f"Theoretical Mean: {(1/p):0.3f} Theoretical Variance: {(1-p)/p**2:.3f}")

# Question 6 f)
from scipy.stats import bernoulli
import numpy as np

def negbin(p, r):
    count = 0
    success = 0
    while success < r:
        count += 1
        if bernoulli.rvs(p) == 1:
            success += 1
    return count

r = 3
N = 10**6
p = 1/6
type(p)

results = [negbin(p, r) for i in range(N)]
mean = np.mean(results)
variance = np.var(results)

print(f"Sample Mean: {mean:.3f} Sample Variance: {variance:.3f}")
print(f"Theoretical Mean: {r/p:.3f} Theoretical Variance: {(r*(1-p))/p**2:.3f}")

# Question 7 b)
from math import comb

def hypergeom(N, K, n, x):
    return (comb(K, x) * comb(N-K, n-x)) / comb(N, n)

N = 30
K = 12
n = 8
x = 4

print(f"The probability of 4 out of 8 members being women is {hypergeom(N, K, n, x):.3f}")

# Question 7 e)
from math import comb

def hypergeom(N, K, n, x):
    return (comb(K, x) * comb(N-K, n-x)) / comb(N, n)

def binom(x, p, n):
    return comb(n, x) * p**x * (1-p)**(n-x)

n = 8
N = 30
p = 0.4
x = 4
K = 12

hyper_results = hypergeom(N, K, n, x)
binom_results = binom(x, p, n)

print(f"Hypergeometric: {hyper_results:.3f} Binomial: {binom_results:.3f}")


"""

# Question 8 c)
from math import factorial
from math import e

def poisson(lam, x):
    return (e**(-lam) * lam**x) / factorial(x)

lam = 6.5

sum = 0
for i in range (0,11):
    sum += poisson(lam, i)

final = 1 - sum
print(f"The probability of more than 10 customers is {final:.4f}")
