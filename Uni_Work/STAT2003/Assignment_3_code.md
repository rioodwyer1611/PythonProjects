# STAT2003 — Assignment 3 Code

**Repo:** `Uni_Work/STAT2003/assignment_3.py`  
→ [[Uni Work/STAT2003/Exam_cheat_sheet|Exam Cheat Sheet]] | [[Uni Work STAT2003]]

---

## Overview

**Pareto** via log-transform, **random sum** (Binomial count × Exponential amounts), normal tail vs $e^{-x^2/2}$ bound.

**Libraries:** `numpy`, `matplotlib`, `scipy.stats.norm`

---

## Question map

| Block | Code pattern | Statistics |
|-------|----------------|------------|
| Q1 | $Z\sim\mathrm{Exp}(\alpha)$, $X=e^Z$; empirical $\bar F(x)$ vs $x^{-\alpha}$ on log-log | Pareto-type tail; $\bar F(x)=x^{-\alpha}$ for $x\ge 1$ |
| Q5 | $N\sim\mathrm{Bin}(m,q)$; $S=\sum_{i=1}^{N} X_i$, $X_i\sim\mathrm{Exp}(\mu)$ iid | Random sum; $E[S]=E[N]E[X]$, Wald |
| (active) | `norm.sf(x)` vs $\exp(-x^2/2)$ | Exact $\Phi$ tail vs crude bound |

---

## Key formulas

**Log-transform:** If $Z\sim\mathrm{Exp}(\alpha)$ and $X=e^Z$, then for $x\ge 1$:
$$\bar F_X(x)=P(X>x)=x^{-\alpha}$$

**Random sum** ($N\perp\{X_i\}$, iid $X_i$):
$$E[S]=E[N]\,E[X],\quad \mathrm{Var}(S)=E[N]\,\mathrm{Var}(X)+\mathrm{Var}(N)\,(E[X])^2$$
(Assignment compares $m=100$, $q=1/5$, $\mu=30$ to $E[S]=600$, $\mathrm{Var}(S)=32400$.)

**Standard normal survival:**
$$P(Z>x)=\Phi(-x)=\texttt{norm.sf(x)}$$

---

## Exam tips

- `exponential(scale=1/α)` in NumPy = rate $\alpha$ in workbook $\mathrm{Exp}(\lambda)$ notation — check parameterisation
- Empty sum when $N=0$ ⇒ $S=0$
- `var()` default $n$ divisor; sample $S^2$ on exam often uses `ddof=1`
