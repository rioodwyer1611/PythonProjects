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
