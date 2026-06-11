# STAT2003 — Exam Cheat Sheet (handwrite, 2 sides)

**Dist table on exam** (`formula_sheet.png`) — pmf/pdf, support, $E[X]$, $\mathrm{Var}$, PGF, MGF. **Not repeated here.**

**Repo:** `Exam_cheat_sheet.md` · [[Workbook_claims]] · [[Workbook_formulas]] · [[Assignment_1_code]] · [[Assignment_2_code]] · [[Assignment_3_code]] · `exam_exclusions.pdf`

**Omitted here:** anything on `formula_sheet.png` (dist pmf/pdf, support, $E$, $\mathrm{Var}$, PGF/MGF) · `exam_exclusions.pdf` (σ-alg, continuity of prob., Claim 30 general quantile, LCG, skew/kurt, Ch 4.6 except $\pi$, ∞-state MC, Cholesky, char. fn., **t/F**, further inequalities, summations 3.50–52). **Still know:** $\mathrm{Exp}$ quantiles (practice exams).

---

## SIDE A

### Probability & counting

De Morgan · $P(A^c)=1-P(A)$ · $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ · inclusion–exclusion for $P(\cup A_i)$ (formula examinable, proof not)

$n!$, $\binom{n}{k}$, multiset $\binom{n+k-1}{k}$ · equiprobable $P(A)=|A|/|\Omega|$

**Conditioning:** $P(A_1\cdots A_n)=P(A_1)P(A_2|A_1)\cdots$ · total prob. $P(A)=\sum_i P(A|B_i)P(B_i)$ · Bayes

**Limits:** $\mathrm{Bin}(n,\lambda/n)\to\mathrm{Poi}(\lambda)$ · $\mathrm{Hyp}\to\mathrm{Bin}$ when $N$ large

---

### RVs, cdf, expectations

$P(a<X\le b)=F(b)-F(a)$ · discrete: watch endpoints

**LOTUS:** $E[h(X)]=\sum h(x)f(x)$ or $\int h(x)f(x)\,dx$ — compute $E[g(X)]$ without new distribution

$E[aX+b]=aE[X]+b$ · $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$ (**not** $a\,\mathrm{Var}+b$) · $\mathrm{Var}(X)=E[X^2]-E[X]^2$

$E[X]=\int_0^\infty \bar F(x)\,dx$ for $X\ge 0$

---

### Markov chains — definitions

**Markov property:** only current state matters —
$$P(X_{n+1}=j \mid X_0,\ldots,X_n=i)=P(X_{n+1}=j \mid X_n=i)=P_{ij}$$

Time-homogeneous: same $P$ every step.

**Stochastic matrix:** $P_{ij}\ge 0$, each **row** sums to $1$ (from $i$ you must go somewhere).

$p^{(0)}$ = initial distribution (row vector). Often start certain in one state: e.g. $p^{(0)}=(1,0,0)$.

---

### Building $P$ from first principles

**Step 1 — list states** (match code labels / story).

**Step 2 — one row per current state $i$:** find all possible $j$ next step + probability.

| Source | How to get $P_{ij}$ |
|--------|---------------------|
| Python `u ~ U(0,1)` | Interval lengths between `if` thresholds |
| Story (“if rainy tomorrow…”) | Given conditional probs directly |
| Continuous driver $V$ | $P_{ij}=\int_{\{\text{rule sends }i\to j\}} f_V(v)\,dv$ |

**Step 3 — check:** every row sums to $1$; impossible transitions get $0$.

**Example (code):** state 0: `u<0.5` stay, `u<0.8` →1, else →2 ⇒ row $(0.5,\,0.3,\,0.2)$.

**Absorbing state $i$:** $P_{ii}=1$, all other entries in row $i$ are $0$.

---

### $n$-step probabilities (two methods)

**A — enumerate paths (workbook / exam favourite)**

$$P(X_{n+1}=j\mid X_n=i)=\sum_k P_{ik}P_{kj}$$

For $n$ steps from $i$ to $j$: sum over all intermediate paths
$$i\to k_1\to k_2\to\cdots\to k_{n-1}\to j$$
(product of edge probs along each path).

**B — matrix**

$$p^{(n)}=p^{(0)}P^n,\qquad (P^n)_{ij}=P(X_n=j\mid X_0=i)$$

**One step by total probability:**
$$P(X_1=j)=\sum_i P(X_0=i)\,P(X_1=j\mid X_0=i)=(p^{(0)}P)_j$$

**Two steps:** $P(X_2=j\mid X_0=i)=\sum_k P_{ik}P_{kj}=(P^2)_{ij}$.

---

### Stationary distribution $\pi$

Long-run row the chain settles to: $\pi=\pi P$, $\sum_j \pi_j=1$, $\pi_j\ge 0$.

**Solve:**
1. Write $\pi P=\pi$ as one equation per state (column view) or $\pi(I-P)=0$.
2. One equation is redundant — replace with $\sum\pi_j=1$.
3. Solve linear system.

**Two-state shortcut** ($P=\begin{pmatrix}a&1-a\\ b&1-b\end{pmatrix}$):

From $\pi_1=a\pi_1+b\pi_2$ and $\pi_1+\pi_2=1$ get $\pi_1=\dfrac{b}{1-a+b}$, $\pi_2=\dfrac{1-a}{1-a+b}$.

**Machine example check:** $P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix}$ ⇒ $\pi=(0.8,\,0.2)$.

**Simulate:** long-run state frequencies $\approx \pi$ (assignment pattern: `choice` from `P[x]`).

---

### Common MC exam types

| Type | Setup |
|------|--------|
| Weather / reliability | Build $P$ from transition rules + given probs |
| Code → $P$ | Read `if u < …` branches per state |
| $P(X_n=j\mid X_0=i)$ | Paths or $P^n$ |
| Find $\pi$ | Solve $\pi P=\pi$, $\sum\pi_j=1$ |
| Gambler’s ruin | States $0,\ldots,N$; $+1$ w.p. $p$, $-1$ w.p. $q$; $0,N$ absorbing; hit prob via paths or MC |
| $m$-step reach | Sum paths of length $\le m$ or $(P^m)_{ij}$ |
| Hitting time from $i$ | If each step hits target w.p. $p$ else stay: $N\sim\mathrm{Geom}(p)$; use $(P^n)_{ij}$ if paths differ |

**Continuous driver $V$:** row $i$ = prob. each rule fires ($P(V>\text{median})=1/2$, etc.)

**Not examinable:** countably infinite state spaces; most of Ch 4.6 theory beyond limiting $\pi$.

---

### Series & calculus tools

$(a+b)^n=\sum\binom{n}{k}a^{n-k}b^k$ · $\Gamma(n)=(n-1)!$, $\Gamma(z+1)=z\Gamma(z)$ · $B(\alpha,\beta)=\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$

Integration by parts · geometric-series sums (3.50–52) **supplied on exam if needed**

---

### 1D transforms & simulation

Monotone $Y=g(X)$: $f_Y(y)=f_X(g^{-1}(y))\bigl|\frac{d}{dy}g^{-1}(y)\bigr|$

$X=F^{-1}(U)$, $U\sim U(0,1)$ · $\mathrm{Exp}$: $X=-\frac{1}{\lambda}\log U$

**$\mathrm{Exp}(\lambda)$ quantiles:** $q_p=-\ln(1-p)/\lambda$ · median $m=\ln(2)/\lambda$ · $P(V>m)=1/2$

Standardise: $Z=(X-\mu)/\sigma\sim N(0,1)$ for table $\Phi$

---

## SIDE B

### Independence & joints

Indep $\Leftrightarrow$ joint density factors · $E[\sum b_i X_i]=\sum b_i E[X_i]$ always · $\mathrm{Var}(\sum b_i X_i)=\sum b_i^2\mathrm{Var}(X_i)$ only if indep

$\mathrm{Var}(aX+bY)=a^2\sigma_X^2+b^2\sigma_Y^2+2ab\,\mathrm{Cov}(X,Y)$ · $\mathrm{Cov}(X+Y,X-Y)=\sigma_X^2-\sigma_Y^2$

Joint normal linear $(A,B)$: indep $\Leftrightarrow$ $\mathrm{Cov}(A,B)=0$ (solve $a$ from $\mathrm{Cov}=0$)

$f_{Y|X}=f_{X,Y}/f_X$ · $\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]$, $\rho=\mathrm{Cov}/(\sigma_X\sigma_Y)$

Joint normal + $\rho=0$ $\Rightarrow$ independent (special case) · indep normals $\Rightarrow$ affine combo is normal

**Bivariate normal conditional:**
$(Z_2|Z_1=z_1)\sim N\!\left(\mu_2+\rho\frac{\sigma_2}{\sigma_1}(z_1-\mu_1),\,\sigma_2^2(1-\rho^2)\right)$

**Tower / total variance:** $E[E[Y|X]]=E[Y]$ · $\mathrm{Var}(Y)=E[\mathrm{Var}(Y|X)]+\mathrm{Var}(E[Y|X])$

---

### Order stats & convolution

$f_{(k)}(x)=\frac{n!}{(k-1)!(n-k)!}F(x)^{k-1}(1-F(x))^{n-k}f(x)$

iid $U(0,1)$: $X_{(k)}\sim\mathrm{Beta}(k,n-k+1)$

$f_{X+Y}(z)=\int f_X(x)f_Y(z-x)\,dx$ · min of $n$ iid $\mathrm{Exp}(\lambda)$ $\sim \mathrm{Exp}(n\lambda)$

**Max of $n$ iid $\mathrm{Exp}(\lambda)$:** $F_{(n)}(x)=(1-e^{-\lambda x})^n$ · $E[X_{(1)}]+\cdots+E[X_{(n)}]=n/\lambda$ (sum of order stats = sum of sample)

**2D change $(X,Y)\to(Z,W)$:** inverse map, $|J|$, support from $X,Y\ge 0$ · $f_{W|Z}=f_{Z,W}/f_Z$

Multivariate: $f_Z=f_X/|\det J_{g^{-1}}|$ · $Z=AX$: $\mu_Z=A\mu$, $\Sigma_Z=A\Sigma A^\top$

---

### Inequalities, samples, limits

**Markov (from first principles):** for $Y\ge 0$, $Y\ge a\mathbf{1}_{Y\ge a}$ $\Rightarrow$ $E[Y]\ge aP(Y\ge a)$; or integral bound on $\{Y\ge a\}$. Apply to $Y=(X-\mu)^2$ for Chebyshev.

Chebyshev: $P(|X-\mu|\ge k\sigma)\le 1/k^2$ · chain: Markov $\Rightarrow$ Cheb $\Rightarrow$ WLLN

$E[\bar X_n]=\mu$, $\mathrm{Var}(\bar X_n)=\sigma^2/n$ · $E[S^2]=\sigma^2$ with divisor $n-1$

iid normal: $\bar X_n\sim N(\mu,\sigma^2/n)$ · $\bar X_n\perp S^2$ · $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$ (Claims 53–54; not $t$/ $F$)

WLLN / SLLN · CLT: $\frac{S_n-n\mu}{\sigma\sqrt{n}}\xrightarrow{d}N(0,1)$ · $\mathrm{Bin}(n,p)\approx N(np,npq)$ if $np,nq>5$

$\bar X_n\approx N(\mu,\sigma^2/n)$, $S_n\approx N(n\mu,n\sigma^2)$ for large $n$

---

### PGF / MGF / branching

**Algebra only** (per-dist $G$, $M$ on exam table): $E[X]=G'(1)$ or $M'(0)$ · $\mathrm{Var}$ from 2nd deriv + C12 · indep sum $\Rightarrow$ multiply $G$ or $M$

**Branching:** $G(z)=\sum p_k z^k$; $\mu=G'(1)$; $\mathrm{Var}=G''(1)+G'(1)-[G'(1)]^2$; extinction $q$ = smallest root of $G(q)=q$ in $[0,1]$; $\mu<1\Rightarrow q=1$; $E[Z_n]=\mu^n$

**Wald:** $E[\sum_{i=1}^N X_i]=E[N]E[X]$ if $N\perp$ iid summands

---

### Exam extras (not on dist table)

$\mathrm{Geom}$: memoryless; $P(X>k)=q^k$; $E[X|X>k]=k+\frac{1}{p}$ · shifted: $N=n_0-1+W$, $W\sim\mathrm{Geom}(p)$

**Truncated geom (cap $K$):** $P(N=k)=q^{k-1}p$ for $k<K$; $P(N=K)=q^{K-1}$ (loop stops at cap)

**Python PMF:** case-split (`if u1<…`) → law of total probability; combine branches

**Cauchy:** no mean, no MGF, CLT fails

---

### Python

`rvs`/`pmf`/`sf` + params · **`trans_matrix[x]`** = row of $P$ · `choice(..., p=row)` simulates MC step · `density=True` vs pdf · `ddof=1` for $S^2$
