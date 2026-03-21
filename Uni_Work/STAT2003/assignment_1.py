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
sum([trial() for _ in range(N)])/N

"""
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

print("The outcome of n = " + str(n) + " and r = " + str(r) + " is " + str(probability_of_b(n,r)))

"""

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

print(str(lowest_r) + " is the minimal value of r. The value of P(B) at this r is " + str(probability_of_b(n, lowest_r)))
print("To check, the value previous is " + str(probability_of_b(n, lowest_r-1)))

# Question 5 c)
from math import comb

def pmf(x):
    return comb(20,x) * (1/4)**x * (3/4)**(20-x)

probability = 0

for i in range(0, 21):
    probability += pmf(i)

print(probability)