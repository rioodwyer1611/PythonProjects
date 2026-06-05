# STAT2003 — Workbook Claims (examinable, compact)

**Source:** `main(1).pdf` · **Table:** `formula_sheet.png` · **Exclusions:** `exam_exclusions.pdf`  
→ [[Exam_cheat_sheet]] · [[Uni Work STAT2003]] · **54 claims** (omit 3, 13, 14, 30 — see bottom) · † = proof not examinable

---

**1 · De Morgan** — $(A\cup B)^c=A^c\cap B^c$, $(A\cap B)^c=A^c\cup B^c$; same with $\bigcup/\bigcap$ over $\{A_i\}$.

**2 · Probability axioms (events)** — $P(\emptyset)=0$ · $A\subseteq B\Rightarrow P(A)\le P(B)$ · $P(A)\le1$ · $P(A^c)=1-P(A)$ · $P(A\cup B)=P(A)+P(B)-P(A\cap B)$.

**4 · Inclusion–exclusion ($n$ events)**† — $P(\bigcup_i A_i)=\sum_k(-1)^{k+1}\sum_{i_1<\cdots<i_k}P(A_{i_1}\cap\cdots\cap A_{i_k})$.

**5 · CDF basics** — $0\le F\le1$ · $F$ non-decreasing · $F(x)\to0,-\infty$; $F(x)\to1,+\infty$.

**6 · Interval prob.** — $P(a<X\le b)=F(b)-F(a)$.

**7 · CDF right-continuous**† — $\lim_{h\downarrow0}F(x+h)=F(x)$.

**8 · LOTUS (discrete)**† — $E[h(X)]=\sum_x h(x)f(x)$.

**9 · Linear $E$** — $E[aX+b]=aE[X]+b$.

**10 · Linear $\mathrm{Var}$** — $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$.

**11 · Additive $E$ (one r.v.)** — $E[g(X)+h(X)]=E[g(X)]+E[h(X)]$.

**12 · Variance formula** — $\mathrm{Var}(X)=E[X^2]-E[X]^2$.

**15 · Poisson limit** — $\mathrm{Bin}(n,\lambda/n)\xrightarrow{n\to\infty}\mathrm{Poi}(\lambda)$ (pmf limit).

**16 · Hyp $\to$ Bin**† — $\mathrm{Hyp}(N,K,n)\approx\mathrm{Bin}(n,p)$ as $N\to\infty$, $K/N\to p$.

**17 · Product rule (events)** — $P(A_1\cdots A_n)=P(A_1)P(A_2|A_1)\cdots P(A_n|A_1\cdots A_{n-1})$.

**18 · Total probability** — partition $\{B_i\}$: $P(A)=\sum_i P(A|B_i)P(B_i)$.

**19 · Bayes** — $P(B_j|A)=\dfrac{P(A|B_j)P(B_j)}{\sum_i P(A|B_i)P(B_i)}$.

**20 · Geom memoryless** — $P(X>s+t\mid X>s)=P(X>t)$.

**21 · MC distribution** — $p^{(n)}=p^{(0)}P^n$.

**22 · Stationary $\pi$** — $\pi=\pi P$, $\pi_j\ge0$, $\sum_j\pi_j=1$ (limiting dist.; Ch 4.6 only this part examinable).

**23 · LOTUS (continuous)** — $E[h(X)]=\int h(x)f(x)\,dx$.

**24 · $E$ / $\mathrm{Var}$ (continuous)** — same as Claims 9–12 (linearity, additivity, $\mathrm{Var}(aX+b)$, computational $\mathrm{Var}$).

**25 · Standardise normal** — $X\sim N(\mu,\sigma^2)\Rightarrow (X-\mu)/\sigma\sim N(0,1)$.

**26 · Tail-sum $E[X]$** — $X\ge0$: $E[X]=\int_0^\infty P(X>x)\,dx$.

**27 · Exp memoryless** — $P(X>x+y\mid X>x)=P(X>y)$.

**28 · 1D transform**† — $Y=g(X)$ monotone: $f_Y(y)=f_X(g^{-1}(y))\,|d g^{-1}/dy|$.

**29 · Inverse transform (continuous $F$)** — $U\sim U(0,1)$, $X=F^{-1}(U)\Rightarrow X\sim F$.

**31 · Markov ineq.** — $X\ge0$: $P(X\ge a)\le E[X]/a$.

**32 · Chebyshev** — $P(|X-\mu|\ge k\sigma)\le 1/k^2$.

**33 · Moment hierarchy**† — finite $E[|X|^s]\Rightarrow$ finite $E[|X|^r]$ for all $r<s$.

**34 · Independence (discrete)** — joint pmf = product of marginals.

**35 · Independence (continuous)** — joint pdf = product of marginals.

**36 · Linearity of $E$** — $E[\sum b_i X_i]=\sum b_i E[X_i]$ (no indep. needed).

**37 · Product $E$** — indep.: $E[X_1\cdots X_n]=E[X_1]\cdots E[X_n]$.

**38 · Correlation** — $-1\le\rho\le1$.

**39 · $\mathrm{Var}$ of indep. sum** — $\mathrm{Var}(\sum b_i X_i)=\sum b_i^2\mathrm{Var}(X_i)$ when indep.

**40 · Normal: uncorrelated $\Rightarrow$ indep.** — joint normal + $\mathrm{Cov}=0\Rightarrow$ independent.

**41 · Normal linear combo**† — indep. $N(\mu_i,\sigma_i^2)$: $\sum b_i X_i\sim N(\sum b_i\mu_i,\sum b_i^2\sigma_i^2)$.

**42 · Tower** — $E[E[Y|X]]=E[Y]$.

**43 · Total variance** — $\mathrm{Var}(Y)=E[\mathrm{Var}(Y|X)]+\mathrm{Var}(E[Y|X])$.

**44 · Multivariate transform** — $\mathbf{Z}=g(\mathbf{X})$ invertible: $f_{\mathbf{Z}}(\mathbf{z})=f_{\mathbf{X}}(\mathbf{x})/|\det J|$.

**45 · Linear map $\boldsymbol{\mu},\Sigma$** — $\mathbf{Z}=A\mathbf{X}$: $\boldsymbol{\mu}_{\mathbf{Z}}=A\boldsymbol{\mu}_{\mathbf{X}}$, $\Sigma_{\mathbf{Z}}=A\Sigma_{\mathbf{X}}A^\top$.

**46 · $k$-th order stat.** — $f_{(k)}(x)=\dfrac{n!}{(k-1)!(n-k)!}F^{k-1}(1-F)^{n-k}f$.

**47 · PGF uniqueness** — same PGF $\Leftrightarrow$ same distribution (non-neg. integer r.v.s).

**48 · MGF uniqueness** — $M$ finite near $0$ $\Rightarrow$ fixes distribution.

**49 · PGF of sum** — indep.: $G_{X_1+\cdots+X_n}=\prod_i G_{X_i}$.

**50 · MGF of sum** — indep.: $M_{X_1+\cdots+X_n}=\prod_i M_{X_i}$.

**51 · Sample mean** — iid: $E[\bar X_n]=\mu$, $\mathrm{Var}(\bar X_n)=\sigma^2/n$.

**52 · Sample variance** — iid: $E[S^2]=\sigma^2$ ($n-1$ divisor).

**53 · Normal $\bar X_n$** — iid $N(\mu,\sigma^2)$: $\bar X_n\sim N(\mu,\sigma^2/n)$.

**54 · Normal $S^2$** — iid $N(\mu,\sigma^2)$: $\bar X_n\perp S^2$; $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$ (not $t$/ $F$).

**55 · WLLN** — iid, finite $\mathrm{Var}$: $P(|\bar X_n-\mu|\ge\varepsilon)\to0$.

**56 · SLLN** — iid, $E[X_i]=\mu$: $P(\lim \bar X_n=\mu)=1$.

**57 · CLT** — iid, finite $\sigma^2$: $(S_n-n\mu)/(\sigma\sqrt{n})\xrightarrow{d}N(0,1)$; large $n$: $S_n\approx N(n\mu,n\sigma^2)$, $\bar X_n\approx N(\mu,\sigma^2/n)$.

**58 · Normal approx. Bin** — $X\sim\mathrm{Bin}(n,p)$: $P(X\le k)\approx\Phi\bigl((k-np)/\sqrt{npq}\bigr)$; need $np,nq>5$.

---

## Omitted

| # | Why |
|---|-----|
| 3 | Excluded: continuity of probability |
| 13, 14 | On exam table: $\mathrm{Geom}$, $\mathrm{Poi}$ means |
| 30 | Excluded: general quantile / inverse transform |

**Use exam table for:** all dist. pmf/pdf, support, $E$, $\mathrm{Var}$, PGF/MGF. **Not studied:** $\sigma$-alg, LCG, skew/kurt, Cholesky, char. fn., $\infty$-state MC, Ch 4.6 except $\pi$, $t$/ $F$, summations 3.50–52 (given if needed).
