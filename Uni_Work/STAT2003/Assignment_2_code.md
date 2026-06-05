# STAT2003 — Assignment 2 Code

**Repo:** `Uni_Work/STAT2003/assignment_2.py`  
→ [[Uni Work/STAT2003/Exam_cheat_sheet|Exam Cheat Sheet]] | [[Uni Work STAT2003]]

---

## Overview

Custom discrete RVs, mixtures, **Bayes** with Binomial likelihood, **Markov chain** simulation, gambler’s ruin, median via cdf, **inverse transform** for Exponential.

**Libraries:** `numpy`, `matplotlib`, `scipy.stats.binom`

---

## Question map

| Block | Code pattern | Statistics |
|-------|----------------|------------|
| Q1D | `choice(1..N, p_i\propto i)`; mean/var vs $(2N+1)/3$, $(N-1)(N+2)/18$ | Weighted discrete on $\{1,\ldots,N\}$ |
| Q1E | `geometric` then `choice` within block; histogram `density=True` | Mixture / conditional generation |
| Q2D | For $k=0..5$: `binom.pmf(k,n,p)$ for $Y\mid D$, $Y\mid D^c$; Bayes ratio | $P(D\mid Y=k)=\frac{P(Y=k\mid D)P(D)}{P(Y=k\mid D)P(D)+P(Y=k\mid D^c)P(D^c)}$ |
| Q3e | `trans_matrix`; `choice` from row $x$; long-run state counts | Stationary $\pi$ ≈ simulated frequencies |
| Q4c | Gambler: $+1$ w.p. $p$, $-1$ w.p. $1-p$ until $0$ or $N$ | Random walk hitting prob (MC) |
| Q5d | Bisection on $F(x)=x^3(4-3x)$ until $F(x)=0.5$ | Median = $F^{-1}(0.5)$ |
| Q8d | `Y = -(1/\lambda)\log(U)`, $U\sim U(0,1)$; hist vs $\lambda e^{-\lambda y}$ | Inverse transform: $\mathrm{Exp}(\lambda)$ |

---

## Key formulas

**Bayes (partition $D, D^c$):**
$$P(D\mid Y=k)=\frac{P(Y=k\mid D)\,P(D)}{P(Y=k\mid D)\,P(D)+P(Y=k\mid D^c)\,P(D^c)}$$

**Markov:** row $i$ of $P$ = $P(X_{n+1}=\cdot\mid X_n=i)$; rows sum to $1$.

**Inverse transform (continuous, strictly increasing $F$):**
$$X=F^{-1}(U),\quad U\sim U(0,1)$$
For $\mathrm{Exp}(\lambda)$: $F(x)=1-e^{-\lambda x}$ ⇒ $X=-\frac{1}{\lambda}\log U$.

---

## Exam tips

- Read `if u < t` thresholds → build transition matrix row by row
- `density=True` on histogram ⇒ compare to **pdf**, not pmf
- `binom.pmf(k, n, p)` = $Y\sim\mathrm{Bin}(n,p)$
