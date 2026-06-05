# STAT2003 — Assignment 1 Code

**Repo:** `Uni_Work/STAT2003/assignment_1.py`  
→ [[Uni Work/STAT2003/Exam_cheat_sheet|Exam Cheat Sheet]] | [[Uni Work STAT2003]]

---

## Overview

Discrete probability: simulation, inclusion–exclusion, Binomial tail, Geometric/Negative Binomial simulation, Hypergeometric vs Binomial, Poisson tail.

**Libraries:** `random`, `math.comb`, `scipy.stats.bernoulli`, `numpy`  
**Note:** `np.var` uses divisor $n$ (population), not $n-1$.

---

## Question map

| Block | Code pattern | Statistics |
|-------|----------------|------------|
| Q2c | `trial()`: 3× `randint(0,8)`, success if sum $=8$; MC over $10^6$ | Equiprobable simulation |
| Q4d–e | `probability_of_b(n,r)` = $\sum_{k=0}^n (-1)^k \binom{n}{k}\bigl(\frac{n-k}{n}\bigr)^r$ | Inclusion–exclusion (occupancy) |
| Q5c | Sum `comb(20,x)(1/4)^x(3/4)^{20-x}` for $x=10,\ldots,20$ | $X\sim\mathrm{Bin}(20,\tfrac14)$, $P(X\ge 10)$ |
| Q6e | `geom(p)`: loop until `bernoulli.rvs(p)==1` | $\mathrm{Geom}(p)$ on $\{1,2,\ldots\}$; check $\bar x\approx 1/p$, $s^2\approx (1-p)/p^2$ |
| Q6f | `negbin(p,r)`: trials until $r$ successes | $\mathrm{NegBin}(r,p)$; mean $r/p$ |
| Q7b | `hypergeom(N,K,n,x)` via `comb` | $\mathrm{Hyp}(N,K,n)$ pmf |
| Q7e | Compare `hypergeom` vs `binom(x,p,n)` | $K/N\approx p$ ⇒ Hyp $\approx$ Bin |
| Q8c | `poisson(lam,x)`; sum $x=0..10$, $1-\sum$ | $X\sim\mathrm{Poi}(\lambda)$, $P(X>10)$ |

---

## Key formulas (from this assignment)

**Inclusion–exclusion (occupancy):**
$$P(B)=\sum_{k=0}^{n}(-1)^k\binom{n}{k}\left(\frac{n-k}{n}\right)^r$$

**Hypergeometric pmf:**
$$P(X=x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}$$

**Geometric (trials until first success):**
$$P(X=k)=(1-p)^{k-1}p,\quad E[X]=\frac{1}{p},\quad \mathrm{Var}(X)=\frac{1-p}{p^2}$$

---

## Exam tips

- `comb(n,k)` ↔ $\binom{n}{k}$
- Simulated `geom` / `negbin` = understand distribution, not memorise loop
- Compare sample mean/var to **theoretical** before trusting MC
