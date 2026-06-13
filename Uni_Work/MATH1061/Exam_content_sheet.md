# MATH1061 — Exam Content Sheet

**Course:** Discrete Mathematics (UQ) · **Also:** MATH7861  
**Sources:** Lecture notes (Lectures 2–39), final exams 2020–2025, exercise solutions Weeks 2–13, `Glossary.pdf`  
**Exam format:** 120 min, closed book (recent), FX-82 calculator, formula sheet attached (logical laws, set identities, quotient-remainder, unique factorisation, Schröder–Bernstein, binomial theorem)  
**2026 final structure:** 13 questions (Lecture 39 revision)

---

## 2026 assessability (Lecture 39 revision — prioritise this)

| In scope | Out of scope / de-emphasised |
|----------|------------------------------|
| Logic: validity, equivalences, truth tables | — |
| Quantified & conditional statements + negations | — |
| GCD, prime factorisation, Euclidean algorithm | — |
| (Recursive) sequences, standard & strong induction | **WOP proofs not assessable** (Lecture 15) — know definition only |
| Sets: $\cup,\cap,-,\times,\mathcal{P}(A)$; prove equalities | — |
| Functions: domain, codomain, image, fibre/preimage, injective, surjective, composition | — |
| Relations: equivalence relations & partial orders | — |
| Cardinality: “same size?” via bijections | **No uncountable business** (Lec 23 marked not assessable; no Cantor diagonal) |
| Groups & subgroups; Cayley tables; examples below | **No proving associativity** from axioms |
| Rings $(\mathbb{Z}_n,+,\times)$; fields $(F,+,\times,0,1)$; solve linear equations in $\mathbb{Z}_p$ | Ring not named in glossary — concept still needed for “why not a field?” |
| Binomial coefficients; counting; probability | — |
| Inclusion–exclusion (2–3 sets); pigeonhole | General $n$-set I–E proof by induction (challenge only) |
| Graphs: Euler trail/circuit, trees, adjacency matrix, handshake | **Matrix multiplication** not assessable (Lec 37) |

**Group examples to know:** $(\mathbb{Z}_n,+)$, $(\mathbb{Z}_p\setminus\{0\},\times)$ for prime $p$, $(\mathbb{Q},+)$, $(\mathbb{R},+)$.

**Past exams (pre-2026)** also tested: group isomorphism, onto-function counting, Schröder–Bernstein proofs — treat as bonus if time permits.

---

## Lecture map (weeks → lectures → topics)

| Week | Lectures | Topics |
|------|----------|--------|
| 1 | 2–3 | Logical forms ($\sim,\land,\lor$); truth tables; logical equivalence laws |
| 2 | 4–6 | Conditionals, contrapositive, negation of $p\to q$; arguments & rules of inference; quantifiers ($\forall,\exists$); number sets $\mathbb{N},\mathbb{Z},\mathbb{Q},\mathbb{R}$ |
| 3 | 7–9 | Multiple quantifiers & negation; direct proofs & counterexamples; proof by contradiction |
| 4 | 10–12 | Proof by contraposition; divisibility & unique factorisation; modular arithmetic, quotient–remainder, floor/ceiling |
| 5 | 13–15 | GCD, LCM, Euclidean algorithm; sequences & summation notation; induction; **strong induction & well-ordering** |
| 6 | 16–18 | Recursive definitions (sequences **and sets**); solving recursive relations; set theory notation ($\in$ vs $\subseteq$) |
| 7 | 19–21 | Set proofs (element method, logical equivalence, set identities); functions 1-1/onto/inverse; composition |
| 8 | 22–24 | Cardinality & bijections (Lec 22); countable/uncountable intro (**Lec 23 — not assessable**); relations: reflexive, symmetric, transitive; arrow diagrams |
| 9 | 25–27 | Equivalence relations & partitions; partial & total orders, **Hasse diagrams**, comparability; groups intro via $\mathbb{Z}_n$ |
| 10 | 28–30 | Group properties; subgroups; fields; introduction to counting (permutations & selections) |
| 11 | 31–33 | Counting selections; probability from counting; binomial theorem & coefficients |
| 12 | 34–36 | Inclusion–exclusion; pigeonhole; graph definitions (loops/multiedges allowed); walks/trails/circuits; Euler conditions |
| 13 | 37–39 | Adjacency matrix (read/write graph); **trees**; final exam revision |

---

## 1. Logic & proofs

### 1.1 Propositional logic

**Statement:** declarative sentence with truth value T or F.

| Connective | Symbol | True when |
|------------|--------|-----------|
| Negation | $\sim p$, $\neg p$ | opposite of $p$ |
| Conjunction | $p \land q$ | both true |
| Disjunction | $p \lor q$ | at least one true |
| Conditional | $p \to q$ | false only if $p$ true and $q$ false |
| Biconditional | $p \leftrightarrow q$ | same truth value |

**Key equivalences** (on formula sheet — name laws in proofs):
- $p \to q \equiv \sim p \lor q$
- Contrapositive: $p \to q \equiv \sim q \to \sim p$ (only equivalent form besides original — converse/inverse are **not** equivalent)
- **Negation of conditional:** $\sim(p \to q) \equiv p \land \sim q$ (common exam trap)
- De Morgan: $\sim(p \land q) \equiv \sim p \lor \sim q$; $\sim(p \lor q) \equiv \sim p \land \sim q$
- Tautology $t$ (always T), contradiction $c$ (always F)

**Necessary / sufficient** (Lecture 4): for $p \to q$, $p$ sufficient for $q$; $q$ necessary for $p$. Biconditional $p \leftrightarrow q$ = necessary and sufficient.

**English ↔ logic:** “$q$ if $p$”, “$p$ only if $q$”, “$p$ is sufficient for $q$”, “$q$ is necessary for $p$” all express $p \to q$.

**Exam tasks:** prove tautology (truth table or named laws); simplify expressions; custom connectives (e.g. NAND, NOR-style $\uparrow\downarrow$ in 2024).

### 1.2 Arguments

**Valid:** whenever all premises true, conclusion must be true.  
**Invalid:** find valuation where premises true but conclusion false.

**Rules of inference** (Lecture 5 — on formula sheet): Modus Ponens, Modus Tollens, Generalization, Specialization, Conjunction, Elimination, Transitivity, Proof by Division into Cases, Contradiction Rule.

**Exam pattern:** state valid/invalid + counterexample valuation (2025 S2 Q1, 2024 Q1b). Always justify.

### 1.3 Quantifiers

| Symbol | Meaning |
|--------|---------|
| $\forall$ | for all |
| $\exists$ | there exists |

**Negation rules** (swap quantifier + negate predicate):
- $\sim(\forall x\, P(x)) \equiv \exists x\, \sim P(x)$
- $\sim(\forall x\, (P \to Q)) \equiv \exists x\, (P \land \sim Q)$
- $\sim(\exists x\, P(x)) \equiv \forall x\, \sim P(x)$

**Multiple quantifiers** (Lecture 7): negate inside-out — e.g. $\sim(\forall x\,\exists y\, P(x,y)) \equiv \exists x\,\forall y\, \sim P(x,y)$. Order matters: $\forall x\,\exists y$ vs $\exists y\,\forall x$ are different.

**Number-set convention (Lecture 6):** course uses $\mathbb{N}=\{0,1,2,\ldots\}$. Exams sometimes use $\mathbb{N}=\{1,2,3,\ldots\}$ or $\mathbb{Z}^+$ — **read the question**.

**Exam tasks:** negate quantified statements; decide truth with counterexample or proof (2025 S1 Q2, 2025 S2 Q2).

### 1.4 Proof methods

| Method | When to use |
|--------|-------------|
| **Direct** | assume hypothesis, derive conclusion |
| **Contrapositive** | prove $\sim Q \to \sim P$ instead of $P \to Q$ |
| **Contradiction** | assume negation, reach impossibility |
| **Counterexample** | disprove $\forall$ statement |
| **Cases** | split by parity, remainder mod $n$, etc. |

**Exam examples:** irrationality proofs (2023 Q3); perfect squares + prime factorisation (2024 Q2a); divisibility true/false (2025 S1 Q3).

---

## 2. Integers, divisibility & modular arithmetic

### 2.1 Number sets (standard domains)

**Containment chain:** $\mathbb{N} \subseteq \mathbb{Z} \subseteq \mathbb{Q} \subseteq \mathbb{R}$ (using MATH1061’s $\mathbb{N}$ below).

| Symbol | Name | Definition | Examples | Exam notes |
|--------|------|------------|----------|------------|
| $\mathbb{Z}$ | **Integers** | $\{\ldots,-3,-2,-1,0,1,2,3,\ldots\}$ — numbers with no fractional part | $-7$, $0$, $42$ | Default domain for divisibility, mod arithmetic, even/odd proofs |
| $\mathbb{N}$ | **Natural numbers** (this course) | $\{0,1,2,\ldots\}$ (Lecture 6) | $0$, $1$, $15$ | **Ambiguous in exams** — some papers use $\mathbb{N}=\{1,2,3,\ldots\}$. Check the question before using $0$ |
| $\mathbb{Z}^+$, $\mathbb{Z}_{>0}$ | **Positive integers** | $\{1,2,3,\ldots\}$ | $1$, $7$, $100$ | Lecture 4 uses $\mathbb{Z}_{>0}$ for “positive integers”. Same set as “counting numbers” |
| $\mathbb{Z}_{\ge 0}$ | **Non-negative integers** | $\{0,1,2,\ldots\}$ | $0$, $5$ | **Equals $\mathbb{N}$ in MATH1061** — different symbol, same elements |
| $\mathbb{Q}$ | **Rational numbers** | $\left\{\dfrac{m}{n} \mid m,n \in \mathbb{Z},\, n \ne 0\right\}$ — ratio of two integers | $\dfrac{3}{4}$, $-2=\dfrac{-2}{1}$, $0.75$ | Closed under $+$, $-$, $\times$ (but not division by $0$). Every rational has a decimal that terminates or repeats |
| $\mathbb{R}$ | **Real numbers** | All points on the number line; any decimal expansion (terminating, repeating, or non-repeating) | $1$, $\pi$, $\sqrt{2}$, $-0.1$ | Superset of $\mathbb{Q}$. Used for floor functions, intervals $[a,b]$, cardinality comparisons |
| $\mathbb{R}\setminus\mathbb{Q}$ | **Irrational numbers** | Real numbers that are **not** rational; equivalently cannot be written as $\dfrac{a}{b}$ with $a,b \in \mathbb{Z}$, $b \ne 0$ | $\sqrt{2}$, $\pi$, $e$ | Every real is **either** rational **or** irrational (not both). $\sqrt{2}\in\mathbb{R}\setminus\mathbb{Q}$ is a standard proof-by-contradiction example |

**How they relate:**
- $\mathbb{Z}^+ \subset \mathbb{N} \subseteq \mathbb{Z} \subseteq \mathbb{Q} \subseteq \mathbb{R}$
- $\mathbb{R} = \mathbb{Q} \cup (\mathbb{R}\setminus\mathbb{Q})$ and $\mathbb{Q} \cap (\mathbb{R}\setminus\mathbb{Q}) = \emptyset$
- $\mathbb{Q}\setminus\mathbb{Z}$ = non-integer rationals (e.g. $\frac{1}{2}$); $\mathbb{R}\setminus\mathbb{Z}$ includes irrationals and non-integer rationals

**Typical quantifier domains in exams:**
- “for all integers $a,b$” → $a,b \in \mathbb{Z}$
- “for all positive integers” → $a,b \in \mathbb{Z}^+$ or $\mathbb{Z}_{>0}$
- “for all real numbers $x$” → $x \in \mathbb{R}$
- “$rs \in \mathbb{Q}$” → product is rational (2025 S1 Q2b)

### 2.2 Divisibility & parity

- $k \mid n$: $n = ak$ for some integer $a$ ($k$ is a **factor** / **divisor** of $n$; $n$ is a **multiple** of $k$)
- **Even:** $n = 2k$ for some $k \in \mathbb{Z}$; **odd:** $n = 2k+1$. Note $0$ is even
- **Prime:** integer $>1$ whose only positive divisors are $1$ and itself
- **Composite:** integer $>1$ that is not prime ($n=rs$ with $r,s>1$). $0$ and $1$ are neither prime nor composite

### 2.3 Quotient–remainder theorem

For integers $n$, $d \ne 0$: **unique** $q,r \in \mathbb{Z}$ with $n = dq + r$ and $0 \le r < |d|$.

**Exam:** compute quotient/remainder for **negative dividends and divisors** (e.g. $-27$ divided by $6$ → $q=-5$, $r=3$).

### 2.4 GCD, LCM & Euclidean algorithm

$\gcd(a,b)$ = largest $d$ dividing both. $\mathrm{lcm}(a,b)$ = smallest positive multiple of both.

Repeated division: $\gcd(a,b) = \gcd(b, a \bmod b)$ until remainder $0$. **Faster than prime factorisation** for large numbers.

**Key identity:** $\gcd(a,b) \cdot \mathrm{lcm}(a,b) = |ab|$ (2024 Q2d).

### 2.5 Prime factorisation

**Unique factorisation:** every $n>1$ is product of primes, unique up to order.

**Divisors:** if $n = p_1^{e_1}\cdots p_k^{e_k}$ then number of positive divisors is $(e_1+1)\cdots(e_k+1)$ (2025 S1 Q5c).

### 2.6 Modular arithmetic

$a \equiv b \pmod{n}$ iff $n \mid (a-b)$.

**Techniques:**
- Reduce mod $n$ by checking residues $0,1,\ldots,n-1$
- Find multiplicative inverse: $ax \equiv 1 \pmod{m}$ when $\gcd(a,m)=1$

### 2.7 Modular arithmetic → fields

Modular arithmetic is the computational tool; **fields** are the algebraic framework. See **§7.4–7.7** for ring/field definitions, when $\mathbb{Z}_n$ is a field, and how to solve $ax+b=c$ in $(\mathbb{Z}_p,+,\times)$.

**Exam:** solve in $(\mathbb{Z}_{11},+,\times)$, $(\mathbb{Z}_{23},+,\times)$, $(\mathbb{Z}_{53},+,\times)$ (2025 S1 Q10, 2025 S2 Q8, 2023 Q11).

### 2.8 Floor / ceiling

$\lfloor x\rfloor$ = greatest integer $\le x$; $\lceil x\rceil$ = smallest integer $\ge x$.

**Useful bounds:** $\lfloor x\rfloor + \lfloor y\rfloor \le \lfloor x+y\rfloor \le \lfloor x\rfloor + \lfloor y\rfloor + 1$ (equality fails in general).

**Exam:** counterexamples involving $\lfloor x^2\rfloor \ne \lfloor x\rfloor^2$ (2024 Q2b); floor functions as maps $\mathbb{R}\to\mathbb{Z}$ (2025 S2 Q6).

---

## 3. Sequences & induction

### 3.1 Sequences

- **Explicit formula:** $a_n$ directly in terms of $n$
- **Recursive:** base case(s) + rule linking new terms to earlier ones (sequences **and sets**, Lecture 16)
- **Summation:** $\sum_{i=a}^{b} f(i)$ notation (Lecture 14)

**Exam:** compute terms; guess explicit form; prove by induction or verify recurrence directly (2025 S2 Q4: $a_{n+1}=a_n+2n+1 \Rightarrow a_n=n^2$).

### 3.2 Mathematical induction

**Standard:** base case → assume $P(k)$ → prove $P(k+1)$.

**Strong induction:** assume $P(\text{all prior cases})$ → prove $P(k+1)$. Needed when recurrence depends on multiple predecessors (Fibonacci-type).

**Well-ordering principle (WOP)** (Lecture 15) — **know the idea, not assessable as a proof method:**

> Every non-empty subset of $\mathbb{Z}^+$ (or $\mathbb{N}$ if excluding $0$) has a **least element**.

**How it is used in theory:** proof by contradiction — assume a statement fails for some positive integers, let $S$ be the set of counterexamples, take the **smallest** $n \in S$ via WOP, then derive a contradiction (e.g. prime factorisation of integers $>1$ — Lecture 15 p.4, marked **“NOT ASSESSABLE”** on slides).

**Equivalent to:** ordinary induction and strong induction (same logical power).

**What you need for the exam:**
- **Use strong induction** when a recurrence depends on multiple prior terms (Fibonacci-type, Week 6/7 exercises; 2023 Q4)
- **Use standard induction** for single-step claims ($3^n+5 < 4^n$, etc.)
- **Do not** write a full WOP proof unless explicitly asked (no past final has done this; Lecture 15 flags it not assessable)

**Exam topics:** inequalities ($3^n+5 < 4^n$, $3^n < (n+3)!$, Fibonacci bounds); parity; sequences with gcd in recurrence (2023 Q4).

**Checklist:** state base case(s); clear inductive hypothesis with correct bounds; show what you need to prove; algebra in inductive step.

---

## 4. Set theory

### 4.1 Operations

| Operation | Definition |
|-----------|------------|
| $A \cup B$ | elements in $A$ or $B$ |
| $A \cap B$ | elements in both |
| $A - B$ | in $A$ not in $B$ |
| $A^c$ | complement in universal set $U$ |
| $\mathcal{P}(A)$ | set of all subsets |
| $A \times B$ | ordered pairs $(a,b)$ |

**Cardinality:** $|\mathcal{P}(A)| = 2^{|A|}$; $|\mathcal{P}(\mathcal{P}(A))|$ (2024 Q4e).

**Set-builder:** $\{X \mid \text{condition}\}$ e.g. $\{X \mid X \subseteq A \land X \in C\}$ (2025 S1 Q6e).

### 4.2 Proving set equalities (Lecture 19)

Three methods:
1. **Element method:** show $x \in A \Leftrightarrow x \in B$
2. **Logical equivalence** of membership conditions
3. **Set identities** from formula sheet

**Critical distinction:** $x \in A$ (“element of”) vs $B \subseteq A$ (“subset of”). e.g. $\{1,2\} \in \mathcal{P}(\{1,2,3\})$ but $\{1,2\} \notin \{1,2,3\}$.

### 4.3 Set identities (on formula sheet)

Commutative, associative, distributive, De Morgan, absorption, etc.

### 4.4 Partitions

Partition of $S$: non-empty subsets, disjoint, union = $S$.

**Equivalence relation** $\leftrightarrow$ partition into equivalence classes.

---

## 5. Functions

### 5.1 Definitions

$f: X \to Y$: domain $X$, codomain $Y$, rule $f(x)$.

| Property | Definition |
|----------|------------|
| **Injective (1-1)** | $f(a)=f(b) \Rightarrow a=b$ |
| **Surjective (onto)** | $\forall y \in Y\,\exists x: f(x)=y$ |
| **Bijective** | both injective and surjective |

**Image:** $f(S) = \{f(x) : x \in S\}$. **Preimage / fibre:** $f^{-1}(T) = \{x : f(x) \in T\}$ (Lecture 39 uses “fibre”).

### 5.2 Composition

$(f \circ g)(a) = f(g(a))$ — order matters.

### 5.3 Cardinality via functions

- Injective $f: A \to B$ $\Rightarrow$ $|A| \le |B|$
- Surjective $f: A \to B$ $\Rightarrow$ $|A| \ge |B|$
- Bijection $\Rightarrow$ $|A| = |B|$

**Schröder–Bernstein:** $|A|\le|B|$ and $|A|\ge|B|$ $\Rightarrow$ $|A|=|B|$.

### 5.4 Cardinality facts

**Definition (Lecture 22):** $|A|=|B|$ iff bijection $A \to B$ exists.

| Comparison | Equal? | Notes |
|------------|--------|-------|
| $|\mathbb{Z}|$ vs $|\mathbb{Z}\times\mathbb{Z}|$ | Yes | Countable |
| $|\mathbb{Z}|$ vs $|\mathbb{Q}|$ | Yes | Countable |
| $|\mathbb{R}|$ vs $|\mathbb{Q}\times\mathbb{Q}|$ | No | $\mathbb{R}$ uncountable |
| $|[0,1]|$ vs $\mathbb{R}^+$ | Yes | Bijection or Schröder–Bernstein (on formula sheet) |

**2026 scope:** construct/check bijections; compare finite and “same cardinality” — **no Cantor diagonal / deep uncountability proofs** (Lectures 23 & 39).

**Exam:** construct explicit bijection $f(n)=2n$ for evens $\leftrightarrow \mathbb{Z}$ (2025 S2 Q9a); prove $|A|\le|B|$ via injection (2023 Q7).

---

## 6. Relations

### 6.1 Binary relation

$R \subseteq A \times B$; write $aRb$ if $(a,b)\in R$.

### 6.2 Properties (on set $A$)

| Property | Condition |
|----------|-----------|
| **Reflexive** | $\forall a,\, aRa$ |
| **Symmetric** | $aRb \Rightarrow bRa$ |
| **Antisymmetric** | $aRb \land bRa \Rightarrow a=b$ |
| **Transitive** | $aRb \land bRc \Rightarrow aRc$ |

### 6.3 Special relations

- **Equivalence relation:** reflexive + symmetric + transitive → equivalence classes, partition
- **Partial order:** reflexive + antisymmetric + transitive (not every pair comparable)
- **Total order:** partial order + all pairs comparable
- **Comparable:** $aRb$ or $bRa$ (Lecture 26)
- **Hasse diagram:** draw partial order covering relations only (e.g. divisibility on $\{1,2,3,4,6,8,12,24\}$)

**Exam:** prove equivalence (modular conditions, gcd-based, $(a,b)\sim(c,d)$ iff $a+d=b+c$); count classes; test partial order (antisymmetry failure); draw arrow diagrams with given properties (2023 Q9).

### 6.4 Modular equivalence

$a \equiv b \pmod{n}$ defines equivalence on $\mathbb{Z}$ with $n$ classes.

---

## 7. Groups, rings & fields

### 7.1 Algebraic structures — how they build up

One operation → two operations → stronger multiplicative structure:

```
Group (G, *)          one operation, all elements invertible
    ↓
Ring (R, +, ·)        two operations; (R,+) abelian group; × associative with 1;
                      distributive; multiplicative inverses NOT required
    ↓
Field (F, +, ·)       ring where every nonzero element has a multiplicative inverse
```

**MATH1061 note:** “Ring” is **not** in the official glossary, but the idea appears whenever you work in $(\mathbb{Z}_n,+,\times)$. Understanding rings explains why some $\mathbb{Z}_n$ fail to be fields.

| Structure | Example | Additive group? | Every nonzero has × inverse? |
|-----------|---------|-----------------|------------------------------|
| Group | $(\mathbb{Z},+)$ | — (only one op) | N/A |
| Ring | $(\mathbb{Z}_{12},+,\times)$ | Yes | **No** ($2\cdot 6\equiv 0$) |
| Ring | $(\mathbb{Z},+,\times)$ | Yes | **No** ($2$ has no inverse in $\mathbb{Z}$) |
| Field | $(\mathbb{Q},+,\times)$, $(\mathbb{R},+,\times)$ | Yes | Yes |
| Field | $(\mathbb{Z}_p,+,\times)$, $p$ prime | Yes | Yes |

---

### 7.2 Group $(G, *)$

A **group** is a set $G$ with binary operation $*: G \times G \to G$ such that:

1. **Closure:** $a*b \in G$ for all $a,b \in G$
2. **Associativity:** $(a*b)*c = a*(b*c)$
3. **Identity:** $\exists e \in G$ with $e*a = a*e = a$
4. **Inverses:** for each $a \in G$, $\exists a^{-1} \in G$ with $a*a^{-1} = a^{-1}*a = e$

**Abelian (commutative):** $a*b = b*a$ for all $a,b$.

**2026:** assume associativity for standard examples — do not prove from axioms (Lecture 39).

**Examples:** $(\mathbb{Z},+)$, $(\mathbb{Z}_n,+)$, $(\mathbb{Z}_p\setminus\{0\},\times)$ for prime $p$, $(\mathbb{Q},+)$, $(\mathbb{R},+)$.

---

### 7.3 Subgroup test

Non-empty $H \subseteq G$ is a **subgroup** iff:

1. **Closed** under $*$: $a,b \in H \Rightarrow a*b \in H$
2. **Contains identity** $e$
3. **Contains inverses:** $a \in H \Rightarrow a^{-1} \in H$

(Associativity inherited from $G$.)

**Exam:** $\{0,2,4,6\}$ in $(\mathbb{Z}_8,+)$; even integers in $(\mathbb{Z},+)$ (2025 S2 Q9b).

---

### 7.4 Ring $(R, +, \cdot)$

A **ring** is a set $R$ with two operations $+$ and $\cdot$ such that:

1. **$(R,+)$ is an abelian group** (closure, associativity, identity $0$, inverses $-a$, commutativity)
2. **$(R,\cdot)$ is closed and associative** with **multiplicative identity** $1$ (in MATH1061 examples)
3. **Distributivity:** $a\cdot(b+c) = a\cdot b + a\cdot c$ and $(b+c)\cdot a = b\cdot a + c\cdot a$

**Key point:** elements of $R\setminus\{0\}$ need **not** have multiplicative inverses. That is what separates rings from fields.

**Commutative ring:** $a\cdot b = b\cdot a$ for all $a,b$ (all examples in this course).

| Ring | Why it is a ring | Why it is **not** a field |
|------|------------------|---------------------------|
| $(\mathbb{Z},+,\times)$ | Usual integer arithmetic | $2$ has no multiplicative inverse in $\mathbb{Z}$ |
| $(\mathbb{Z}_n,+,\times)$ | Addition and multiplication mod $n$ always work | Fails when $n$ composite — see §7.6 |
| $(\mathbb{Z}_{12},+,\times)$ | Same as above | $2\cdot 6 \equiv 0 \pmod{12}$; $2$ has no inverse mod $12$ (2021 S2 Q10b) |

**Connection to groups:** from a ring you always get $(R,+)$ as an abelian group. The nonzero elements may or may not form a group under $\cdot$.

---

### 7.5 Field $(F, +, \cdot)$

A **field** $(F,+,\cdot)$ (Glossary; Lecture 29) satisfies:

1. **$(F,+)$ is an abelian group** with identity $0$
2. **$(F\setminus\{0\},\cdot)$ is an abelian group** with identity $1$
3. **Distributivity:** $a\cdot(b+c) = a\cdot b + a\cdot c$ for all $a,b,c \in F$

**Notation on exams:** $(\mathbb{Z}_p, +, \times)$ or $(\mathbb{Z}_{11}, +, \times)$ — elements are residue classes $\{[0],[1],\ldots,[p-1]\}$.

**Standard field examples (Lecture 39):**

| Field | Elements | Notes |
|-------|----------|-------|
| $(\mathbb{Q},+,\times)$ | Rationals | Infinite |
| $(\mathbb{R},+,\times)$ | Reals | Infinite |
| $(\mathbb{Z}_p,+,\times)$ | $\{[0],\ldots,[p-1]\}$ mod $p$ | **Finite**; $p$ must be prime |

**From a field you get two groups:**
- $(F,+)$ — all elements
- $(F\setminus\{0\},\times)$ — all nonzero elements

---

### 7.6 When is $\mathbb{Z}_n$ a field?

$(\mathbb{Z}_n,+,\times)$ is always a **ring**. It is a **field if and only if $n$ is prime**.

**Why prime matters:** in $\mathbb{Z}_n$, $[a][b]=[0]$ can happen even when $[a]\ne [0]$ and $[b]\ne [0]$ if $n$ is composite (zero divisors). Then $(\mathbb{Z}_n\setminus\{[0]\},\times)$ cannot be a group.

**Non-field example — $\mathbb{Z}_{12}$** (2021 S2 Q10b):
- $[2]\cdot[6] = [12] = [0]$, but $[2]\ne [0]$ and $[6]\ne [0]$
- So $[2]$ has no multiplicative inverse mod $12$ (if $[2][x]=[1]$, multiply both sides by $[6]$ to get $[0]=[6]$ — contradiction)

**Non-group example — $\mathbb{Z}_{10}\setminus\{[0]\}$** (Week 10 exercise):
- $[2]\cdot[5] = [10] = [0]$ — product of two nonzero residues is zero, so closure fails for multiplication on nonzero elements

**Prime-field example — $\mathbb{Z}_{11}$:**
- Every $[a]\ne [0]$ has a multiplicative inverse (because $\gcd(a,11)=1$)

---

### 7.7 Multiplicative inverses & solving equations

**Multiplicative inverse** of $[a]$ in $\mathbb{Z}_n$: an element $[x]$ such that $[a][x]=[1]$.

**Exists iff** $\gcd(a,n)=1$. For prime $n=p$, every $[a]\ne [0]$ has an inverse.

**Finding inverses:**
- **Guess-and-check** on small exams (hint often given: e.g. $3\times 4\equiv 1 \pmod{11}$)
- **Extended Euclidean algorithm** (if no hint) — find $x$ with $ax+ny=1$, then $[x]$ is the inverse

**Solving linear equations** $ax + b = c$ in field $(\mathbb{Z}_p,+,\times)$ (Lecture 29; 2025 S2 Q8):

1. Rearrange: $ax = c - b$ (subtract $b$ in the field)
2. Reduce RHS mod $p$
3. Multiply both sides by $a^{-1}$ (the inverse of $[a]$)

**Worked example** — solve $3x - 5 = 8$ in $\mathbb{Z}_{11}$:
- $3x \equiv 8+5 \equiv 13 \equiv 2 \pmod{11}$
- Hint: $3\cdot 4 \equiv 1$, so multiply by $4$: $x \equiv 2\cdot 4 = 8 \pmod{11}$

**Exam variant:** solve then find smallest non-negative integer $y$ with same congruence (2025 S1 Q10, 2023 Q11).

**Finding inverse only** (2021 S2 Q10a): find $x$ with $7x \equiv 1 \pmod{11}$ — try $x=8$ since $7\cdot 8=56\equiv 1$.

---

### 7.8 Cayley tables

Read from table for finite groups:
- **Identity:** row/col matching headers
- **Inverse of $a$:** find column/row giving identity
- **Abelian:** table symmetric about diagonal

**Exam:** identity, inverse, subgroup $\langle e\rangle$ (2025 S1 Q9).

---

### 7.9 Isomorphism

Bijective map $\phi: G \to H$ preserving operation: $\phi(a*b)=\phi(a)*\phi(b)$.

**Past exams** (2022, 2024) tested group isomorphism; **2026 revision omits it** — lower priority.

---

### 7.10 Cyclic groups & $\mathbb{Z}/n\mathbb{Z}$

**$\mathbb{Z}/n\mathbb{Z} = \mathbb{Z}_n$** — equivalence classes mod $n$ under addition.

Under **multiplication** mod $n$: $[a]$ has a multiplicative inverse iff $\gcd(a,n)=1$ (2020 Q1: no inverse for $[2]$ in $\mathbb{Z}/4\mathbb{Z}$).

**Units mod $n$:** $\{[a] : \gcd(a,n)=1\}$ form a group under $\times$; size is $\varphi(n)$ (Euler totient — name optional, idea useful).

---

## 8. Counting & probability

### 8.1 Permutations & combinations

- **Permutations:** $n!$ of $n$ distinct objects
- **With repeats:** $\dfrac{n!}{n_1! n_2! \cdots}$ (multinomial)
- **Combinations:** $\binom{n}{k} = \dfrac{n!}{k!(n-k)!}$

### 8.2 Binomial theorem (on sheet)

$$(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$$

**Exam:** coefficient of $x^k$ in $(2x-3)^6$, $(3-x)^6$ — identify $k$, compute $\binom{n}{k} \cdot (\text{coeff})^k \cdot (\text{const})^{n-k}$.

### 8.3 Arrangement restrictions

- **Inclusion–exclusion** for "at least one" / "not adjacent"
- **Complement counting:** total − unwanted
- **Glue method:** treat adjacent letters as one block

**Exam:** KOOKABURRA arrangements + probability A's not adjacent (2025 S2 Q11); line arrangements with "not next to" constraints (2025 S1 Q11c).

### 8.4 Functions counting

- **Injective** from $A$ to $B$, $|A|=m$, $|B|=n$: $n(n-1)\cdots(n-m+1)$ if $m\le n$
- **Onto** from $A$ to $B$: inclusion–exclusion or Stirling numbers (2024 Q8c)

### 8.5 Probability

$$P(E) = \frac{\text{favourable outcomes}}{\text{total outcomes}}$$

Express as fraction in lowest terms or percentage.

### 8.6 Pigeonhole principle

$n+1$ objects in $n$ boxes $\Rightarrow$ some box has $\ge 2$.

**Exam:** pick integers from $\{1,\ldots,99\}$ to guarantee divisibility (2025 S1 Q12); congruence classes mod 11 (2023 Q13).

### 8.7 Inclusion–exclusion (3 sets)

$$|A\cup B\cup C| = |A|+|B|+|C| - |A\cap B| - |B\cap C| - |A\cap C| + |A\cap B\cap C|$$

**Exam:** clean IDs divisible by 2,3,5 (Week 13).

---

## 9. Graph theory

### 9.1 Basic definitions

**Graph** $G=(V,E)$: vertices and edges. Unless “simple graph” is specified, **loops and multiple edges are allowed** (Lecture 35). Simple graph = no loops, no parallel edges.

| Term | Meaning |
|------|---------|
| **Degree** $\deg(v)$ | edges incident to $v$ |
| **Walk** | vertex–edge sequence |
| **Trail** | walk, all edges distinct |
| **Circuit** | trail, start = end |
| **Cycle** | circuit, no repeated internal vertices |
| **Connected** | walk between any two vertices |
| **Tree** | connected + acyclic |

### 9.2 Handshake lemma

$$\sum_{v \in V} \deg(v) = 2|E|$$

**Exam:** find missing degree (2024 Q10a, 2025 S1 Q14); tree degree distribution (2022 Q11a).

### 9.3 Euler trails & circuits

Connected graph:
- **Euler circuit:** all vertices even degree
- **Euler trail:** exactly 0 or 2 odd-degree vertices

**Complete graph $K_n$:** $\binom{n}{2}$ edges; Euler circuit iff $n$ odd (all degrees $n-1$).

### 9.4 Matrices

- **Adjacency matrix** $A_{ij}$ = number of edges between $v_i$ and $v_j$ — **write matrix from graph and draw graph from matrix** (Lecture 37)
- **Incidence matrix** (glossary; less emphasised in lectures)
- **Matrix multiplication of adjacency matrices is NOT assessable** (Lecture 37)

**Exam:** write adjacency matrix; draw from matrix; multigraphs allowed (non-zero diagonal / entries $>1$).

### 9.5 Trees (Lecture 38)

**Definition:** connected graph with **no cycles**.

**Properties:**
- $|V| = |E| + 1$
- Every tree with more than one vertex has a vertex of **degree 1** (leaf)
- Handshake lemma + these facts solve degree-distribution problems (2022 Q11, 2025 S1 Q14)

### 9.6 Other graph facts

- Walk parity / bipartite-style arguments (odd-length walk between partitions — 2024 Q10c)
- Adding edges to achieve Euler circuit: pair up odd-degree vertices
- Assume **connected** when applying Euler theorems unless stated otherwise

---

## 10. Exam question type checklist

**2026 (Lecture 39):** 13 questions mapping to sections 1–13 in the assessability table above.

**Past papers (2020–2025)** — still useful practice:

| Q | Topic | Seen in |
|---|--------|---------|
| 1 | Logical equivalence / tautology / argument validity | Every exam |
| 2 | Quantifier negation + truth | 2022–2025 |
| 3 | Prove/disprove divisibility, integers | 2022, 2025 S1 |
| 4 | Induction (standard or strong) | Every exam |
| 5 | Euclidean algorithm, factorisation, divisors, lcm | 2022, 2024, 2025 |
| 6 | Set operations, power set, Cartesian product | Every exam |
| 7 | Functions: 1-1, onto, image, preimage | 2023–2025 |
| 8 | Cardinality / bijections (no Cantor diagonal in 2026) | 2022, 2024, 2025 S2 |
| 9 | Equivalence / partial order relations | Every exam |
| 10 | Groups: Cayley table, subgroup, solve in $\mathbb{Z}_p$ | 2022, 2023, 2025 |
| 11 | Counting, binomial, probability | 2022, 2023, 2025 |
| 12 | Pigeonhole / inclusion–exclusion | 2023, 2025 S1, Week 13 |
| 13 | Graphs: adjacency matrix, Euler, degrees, trees | 2020, 2022, 2023, 2025 |

---

## 11. Formula sheet items (memorise location, not necessarily content)

Provided on exam — **do not reproduce from memory unless practising**:
- All propositional logical equivalences
- Valid argument forms
- 12 set identities
- Quotient–remainder theorem
- Unique factorisation theorem
- Schröder–Bernstein theorem
- Binomial theorem

---

## 12. Quick glossary (from course glossary)

| Term | One-line definition |
|------|---------------------|
| Tautology | always true statement form |
| Contradiction | always false statement form |
| Predicate | sentence becoming proposition when variables fixed |
| Truth set | domain values making predicate true |
| Bijection | 1-1 and onto; invertible function |
| Countable | same cardinality as subset of $\mathbb{Z}_{\ge 0}$ |
| Uncountable | no bijection to $\mathbb{Z}$ |
| Equivalence relation | reflexive, symmetric, transitive |
| Partition | disjoint non-empty subsets covering set |
| Subgroup | subset of group closed under operation with identity & inverses |
| Ring | $(R,+,\cdot)$: $(R,+)$ abelian group; $\cdot$ associative with $1$; distributive (inverses under $\cdot$ not required) |
| Field | ring where $(F\setminus\{0\},\cdot)$ is also an abelian group |
| Zero divisor | $a,b\ne 0$ but $ab=0$ — prevents $\mathbb{Z}_n$ from being a field when $n$ composite |
| Pigeonhole principle | $n+1$ objects, $n$ boxes $\Rightarrow$ collision |
| Eulerian circuit | circuit using every edge exactly once |
| Tree | connected acyclic graph |

---

## 13. Study priorities (by exam weight)

1. **Logic** — negation, validity, equivalence proofs (guaranteed marks)
2. **Induction** — base case + hypothesis + step structure
3. **Sets & functions** — listing elements, 1-1/onto proofs
4. **Relations** — equivalence proof template (reflexive/symmetric/transitive)
5. **Number theory** — Euclidean algorithm, mod arithmetic, field equations
6. **Counting** — permutations with repeats, binomial coefficients, simple probability
7. **Graphs** — handshake, Euler conditions, adjacency matrix
8. **Groups, rings & fields** — Cayley tables, subgroup checks, $\mathbb{Z}_n$ ring vs field, solve $ax+b=c$ in $\mathbb{Z}_p$

---

*Fact-checked against lecture notes (Lectures 2–39, May 2026). Image-based slides — key assessability from Lecture 39 revision. For printing, open in Obsidian or any Markdown viewer.*
