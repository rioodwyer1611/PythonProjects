---
tags:
  - portfolio
  - university
  - statistics
  - STAT2003
---

# STAT2003 — Workbook Formulas (non-Claims)

**Source:** `main(1).pdf` workbook · **Checked:** `exam_exclusions.pdf` (Sem 1 2026) — nothing below is listed as excluded.

→ [[Workbook_claims]] (54 numbered Claims) · [[Exam_cheat_sheet]] (handwrite subset) · [[Uni Work STAT2003]] · `formula_sheet.png` (dist pmf/pdf, $E$, $\mathrm{Var}$, PGF/MGF — **not repeated here**)

**Scope:** workbook recipes **not** already in [[Workbook_claims]]. Condensed exam-facing versions on [[Exam_cheat_sheet]].

**Bold** = from `practice1_sols.pdf` / `practice2_sols.pdf`, not already in [[Workbook_claims]] or sections below.

---

## Practice exam methods (P1 & P2)

### Distributions & normalisation

**Beta identification** (pdf on `formula_sheet.png`): read off $\alpha,\beta$ from exponent pattern $x^{\alpha-1}(1-x)^{\beta-1}$; normalising constant $c=1/B(\alpha,\beta)$ (not on sheet).

**$k$-th order stat of iid $U(0,1)$:** $U_{(k)}\sim\mathrm{Beta}(k,\,n-k+1)$ — not on sheet (sheet has $\mathrm{Beta}(\alpha,\beta)$ pdf only).

**Gamma $\alpha,\lambda$ from moments** (use $E=\alpha/\lambda$, $\mathrm{Var}=\alpha/\lambda^2$ on sheet): $\lambda=\mu/\sigma^2$, $\alpha=\mu^2/\sigma^2$.

**Median / quantiles (continuous):** solve $F(m)=\tfrac12$; $F(q_p)=p$ (Exp quantiles below — cdf not on sheet).

**Normal symmetry** (not on sheet — sheet gives pdf/MGF only): $Z\sim N(0,1)$ is symmetric about $0$, so $\Phi(-z)=1-\Phi(z)$. Two-sided tail:
$$P(|Z|\ge k)=P(Z\ge k)+P(Z\le -k)=2P(Z\ge k)=2\bigl(1-\Phi(k)\bigr)$$
(e.g. P1 Q3c: $P(|X-12|\ge 6)$ with $\sigma=3$ $\Rightarrow$ $P(|Z|\ge 2)=2(1-\Phi(2))$).

### Python code $\to$ distribution

**Case-split PMF:** $P(X=x)=\sum_{\text{branches}} P(\text{branch})\,P(X=x\mid\text{branch})$ (law of total probability / Claim 18).

**Truncated geometric (cap $K$, exit prob $p$ each trial):** $P(N=k)=q^{k-1}p$ for $k=1,\ldots,K-1$; $P(N=K)=q^{K-1}$ (loop stops at cap).

**Shifted geometric** (start $n_0$, success prob $p$): $N=n_0-1+W$, $W\sim\mathrm{Geom}(p)$;
$$P(N=k)=q^{\,k-n_0}p,\quad k=n_0,n_0+1,\ldots;\qquad E[N]=n_0-1+\frac1p$$
$$P(N>k)=q^{\,k-n_0+1}\;(k\ge n_0);\qquad P(N>s+t\mid N>s)=q^t\;\text{(memoryless)}$$
$$E[N\mid N>s]=s+\frac1p$$

**Discrete PGF/MGF from pmf** (definitions; per-dist forms on sheet): $G(z)=\sum_x z^x P(X=x)$, $M(t)=\sum_x e^{tx}P(X=x)$.

### Markov chains

**Absorbing state $i$:** $P_{ii}=1$, rest of row $0$; loop-until-absorption $\Rightarrow$ count steps.

**Sub-chain as $\mathrm{Geom}(p)$:** if from state $i$ only “stay in $i$” (prob $q$) or “hit target” (prob $p$) until exit.

**MC with continuous driver $V$:** e.g. $P(i\to j)=P(V\in\text{interval})$; Exp thresholds $\Rightarrow$ $P(V>m)=1/2$, $P(V<Q_1)=1/4$, $P(V>Q_3)=1/4$, middle $1/2$.

**Long-run mean of driven r.v.:** $E[V]=\sum_i \pi_i\,E[V\mid\text{state }i]$ (stationary $\pi$ from Claim 22 + tower / Claim 18).

### Order statistics & joints

**Sum of order-stat expectations (iid):** $\displaystyle\sum_{i=1}^{n} E[X_{(i)}]=n\,E[X]$ (e.g. find $E[X_{(n)}]$ given $E[X_{(1)}],E[X_{(2)}]$).

**$\mathrm{Cov}(X+Y,\,X-Y)=\mathrm{Var}(X)-\mathrm{Var}(Y)$** ($\mathrm{Cov}(X,Y)$ terms cancel).

**Joint normal $\Rightarrow$ indep:** need $\mathrm{Cov}(A,B)=0$ *and* $(A,B)$ jointly normal; solve $\mathrm{Cov}(aX+bY,\,cX+dY)=0$ for constants.

**Independence from joint pdf (continuous):** need $f_{X,Y}(x,y)=g(x)h(y)$ **and** support a rectangle (not e.g. $\{-z<w<z\}$).

**2D change $(X,Y)\to(Z,W)$ workflow:** inverse map $\to$ Jacobian $|J|\to$ $f_{Z,W}=f_X f_Y |J|\to$ support from $X,Y\ge0\to$ conditional $f_{W|Z}=f_{Z,W}/f_Z$.

### Branching

**Regimes:** $\mu<1$ subcritical ($q=1$); $\mu=1$ critical; $\mu>1$ supercritical ($q<1$, solve $G(q)=q$).

---

## Ch 2 — Counting

**Factorial / binomial coefficient**
$$n! = n(n-1)\cdots 1, \qquad \binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Four elementary counting cases** ($n$ types, choose $k$):

| Order | Replacement | Count |
|-------|-------------|-------|
| Yes | Yes | $n^k$ |
| Yes | No | $\dfrac{n!}{(n-k)!}$ |
| No | No | $\binom{n}{k}$ |
| No | Yes | $\binom{n+k-1}{k}$ |

**Equiprobable finite space:** $P(A) = |A|/|\Omega|$.

**Pascal's rule:** $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$.

**Binomial theorem:** $(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$.

---

## Ch 3 — Discrete extras

**Geometric tail** (Claim 20 is memoryless property, not this): $P(X>k) = q^k$ for $X\sim\mathrm{Geom}(p)$, $q=1-p$.

**Retrieve pmf from PGF** (Ch 9.3, discrete $X\ge 0$):
$$P(X=x) = \frac{1}{x!}\left.\frac{d^x}{dz^x} G(z)\right|_{z=0}$$

---

## Ch 5 — Continuous extras

**Normal interval probabilities** (apply Claim 25):
$$P(a < X < b) = \Phi\!\left(\frac{b-\mu}{\sigma}\right) - \Phi\!\left(\frac{a-\mu}{\sigma}\right), \qquad X\sim N(\mu,\sigma^2)$$

**Empirical rule:** $P(|X-\mu| < k\sigma) \approx 68\%,\,95\%,\,99.7\%$ for $k=1,2,3$.

**Exponential cdf / quantiles** — pdf, $E$, $\mathrm{Var}$, MGF on sheet; **cdf and quantiles are not**:
$$F(x) = 1 - e^{-\lambda x},\quad \bar F(x)=e^{-\lambda x},\quad q_p = -\frac{\ln(1-p)}{\lambda},\quad m = \frac{\ln 2}{\lambda}$$

**Beta function** $B(\alpha,\beta)=\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ — links sheet Beta pdf to normalising constant (not printed on sheet).

*Excluded:* skewness/kurtosis (5.2); Beta reparametrisation by mean & CV; Beta transformations (5.5).

---

## Ch 7 — Joints, covariance, normals

**Conditional pdf:**
$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

**Covariance** (Claim 38 is bounds on $\rho$ only):
$$\mathrm{Cov}(X,Y) = E[(X-EX)(Y-EY)] = E[XY] - E[X]E[Y]$$
$$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)$$
$$\mathrm{Var}(aX+bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$$
$$\rho_{X,Y} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Multinomial pmf** $(X_1,\ldots,X_k)\sim\mathrm{Mnom}(n,p_1,\ldots,p_k)$, $\sum_i x_i = n$:
$$P(X_1=x_1,\ldots,X_k=x_k) = \frac{n!}{x_1!\cdots x_k!}\, p_1^{x_1}\cdots p_k^{x_k}$$
$$\mathrm{Cov}(X_i,X_j) = -n p_i p_j \quad (i\neq j)$$

**Bivariate normal pdf** $(Z_1,Z_2)$ with means $\mu_i$, variances $\sigma_i^2$, correlation $\rho$:
$$f(z_1,z_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(z_1-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(z_1-\mu_1)(z_2-\mu_2)}{\sigma_1\sigma_2} + \frac{(z_2-\mu_2)^2}{\sigma_2^2}\right]\right)$$

**Bivariate normal conditional** (Claim 40: uncorrelated joint normal $\Rightarrow$ independent):
$$(Z_2 \mid Z_1=z_1) \sim N\!\left(\mu_2 + \rho\frac{\sigma_2}{\sigma_1}(z_1-\mu_1),\; \sigma_2^2(1-\rho^2)\right)$$

**Sum of iid $\mathrm{Exp}(\lambda)$:** $S_n = X_1+\cdots+X_n \sim \mathrm{Gamma}(n,\lambda)$ (MGF product, (7.53)).

**Wald (random sums)** — $N$ independent of iid $X_i$ with mean $\mu$, variance $\sigma^2$:
$$E\!\left[\sum_{i=1}^{N} X_i\right] = E[N]\,\mu$$
$$\mathrm{Var}\!\left(\sum_{i=1}^{N} X_i\right) = \sigma^2 E[N] + \mu^2 \mathrm{Var}(N)$$

*(Uses Claims 42–43 for derivation.)*

---

## Ch 8 — Sums, transforms, order stats

**Convolution** (independent continuous $X,Y$, $Z=X+Y$):
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)\,f_Y(z-x)\,dx = (f_X \star f_Y)(z)$$

**Example — sum of two iid $\mathrm{Exp}(\lambda)$:** $f_Z(z) = \lambda^2 z\, e^{-\lambda z}$, $z>0$ (Gamma$(2,\lambda)$).

**Max / min cdfs** ($n$ iid, common cdf $F$); $k$-th order-stat **pdf** is Claim 46:
$$F_{\max}(z) = F(z)^n, \qquad F_{\min}(y) = 1 - (1-F(y))^n$$

**Min of $n$ iid $\mathrm{Exp}(\lambda)$:** $Y \sim \mathrm{Exp}(n\lambda)$.

**Max of $n$ iid $U(0,1)$:** $Z \sim \mathrm{Beta}(n,1)$, $E[Z] = n/(n+1)$.

**Uniform on triangle** $f_{X,Y}(x,y)=2$, $0\le x\le y\le 1$: marginal $f_X(x)=2(1-x)$; $f_{Z}(z)=z$ on $[0,1]$, $2-z$ on $[1,2]$ for $Z=X+Y$.

*(2D Jacobian and linear maps: Claims 44–45.)*

---

## Ch 9 — PGF / MGF algebra

**From PGF** $G(z)=E[z^X]$:
$$E[X] = G'(1), \qquad \mathrm{Var}(X) = G''(1) + G'(1) - [G'(1)]^2$$

**From MGF** $M(s)=E[e^{sX}]$:
$$E[X^n] = M^{(n)}(0), \qquad E[X]=M'(0), \qquad \mathrm{Var}(X) = M''(0) - [M'(0)]^2$$

**Branching process** — offspring pmf $p_k$, $G(z)=\sum_k p_k z^k$, $\mu=G'(1)$:
$$E[Z_n] = \mu^n$$
Extinction probability $q = P(Z_n=0 \text{ for some } n\ge 1)$ solves $G(q)=q$ on $[0,1]$; smallest root. If $\mu\le 1$ then $q=1$ (unless $p_1=1$); if $\mu>1$ then $q<1$.

*(PGF/MGF of sums: Claims 49–50.)*

*Characteristic / Laplace–Stieltjes transforms (9.2) excluded.*

---

## Ch 10 — Sample statistics (definitions)

$$\bar X_n = \frac{1}{n}\sum_{i=1}^{n} X_i, \qquad S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar X_n)^2$$

*(Means and variances of $\bar X_n$, $S^2$: Claims 51–52; normal-sample dists: Claims 53–54.)*

---

## Ch 4 — MC operational recipes

**Markov property (one step):** $P(X_{n+1}=j \mid X_n=i) = P_{ij}$.

**Path enumeration:** sum products of edge probs over all $i\to\cdots\to j$ paths of length $n$.

**Two-state $\pi$ shortcut** ($P=\begin{pmatrix}a&1-a\\ b&1-b\end{pmatrix}$; general $\pi$: Claim 22):
$$\pi_1 = \frac{b}{1-a+b}, \qquad \pi_2 = \frac{1-a}{1-a+b}$$

*(Multi-step law $p^{(n)}=p^{(0)}P^n$: Claim 21.)*

*Excluded: countably infinite state spaces; most of Ch 4.6 except limiting $\pi$.*

---

## Explicitly omitted (exclusions or elsewhere)

| Item | Where instead |
|------|----------------|
| 54 numbered Claims | [[Workbook_claims]] |
| Dist pmf/pdf, $E$, $\mathrm{Var}$, PGF/MGF per distribution | `formula_sheet.png` |
| $\sigma$-algebras, continuity of probability | `exam_exclusions.pdf` Ch 1 |
| LCG algorithm details | Ch 1.4 excluded |
| Summations (3.50)–(3.52) | supplied on exam if needed |
| General quantile function (Claim 30) | excluded; Exp quantiles above OK |
| Skewness, kurtosis | Ch 5.2 excluded |
| Beta CV reparam, Beta transforms | Ch 5.5 excluded |
| Proof of change-of-variable | Ch 6.1; formula is Claim 28 |
| Further inequalities | Ch 6.3 excluded |
| Moment hierarchy (Claim 33) | proof excluded |
| Cholesky | Ch 8.5 excluded |
| Characteristic functions | Ch 9.2 excluded |
| $t$, $F$ distributions | Ch 10.2 excluded |
