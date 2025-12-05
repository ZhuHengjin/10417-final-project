To achieve your goal—using the similarity $w$ to control the magnitude of the push/pull—you should keep the negative term but **modulate** it, rather than delete it.

$$
h(T,S) = \frac{\exp({g^T(T)'g^S(S)/\tau})}{\exp({g^T(T)'g^S(S)/\tau}) + \frac{N}{M}}
$$

$$
1- h(T,S) 
= \frac{\frac{N}{M}}{\exp({g^T(T)'g^S(S)/\tau}) + \frac{N}{M}}
= \frac{N}{M\exp({g^T(T)'g^S(S)/\tau}) + N}
$$

The origin loss in the CRD paper is:
$$\mathcal{L}_{critic}(h) = \mathbb{E}_{q(T,S|C=1)}[\log h(T,S)] + N \mathbb{E}_{q(T,S|C=0)}[\log(1 - h(T,S))]$$

The improved loss with semantic weighting becomes:

$$
\mathcal{L}_{modified} = \mathbb{E}_{pos}[\log h(T,S)] + N \cdot \mathbb{E}_{neg}[ w(T_i, T_j) \cdot \log(1 - h(T,S)) ]
$$

where $w(T_i, T_j) = 1 + \alpha \cdot \exp(\cos(f_t(i), f_t(j)) / \tau)$

* **If $T_i$ and $T_j$ are very similar ($w$ is high):** The weight increases. This creates a "Hard Negative," punishing the student more if it fails to distinguish these similar items.
* **If $T_i$ and $T_j$ are very different ($w$ is low/negative):** The weight decreases. The loss focuses less on these easy pairs because they are already chemically distinct in the teacher's space.

---




In CRD, the distillation objective is formulated as a lower bound of the mutual information (MI) between teacher and student representations. For a parametric critic function $h(T,S)$, the original contrastive loss is

$$
\mathcal{L}_{\text{critic}}(h)
\mathbb{E}_{q(T,S|C=1)}[\log h(T,S)]
+
N,\mathbb{E}_{q(T,S|C=0)}[\log(1 - h(T,S))],
$$

where $q(T,S|C=1)$ denote the joint distribution and $q(T,S|C=0)$ denote the product-of-marginals distributions for positive and negative pairs, respectively. All negative pairs are treated as identically distributed background samples.

However, the teacher representation typically exhibits non-uniform geometric structure, where negative samples vary substantially in semantic proximity to the anchor. Uniform negative sampling therefore leads to suboptimal MI estimation because the critic allocates equal weight to both informative (nearby) and trivial (far-apart) negatives.

Intuitively, negatives that are more semantically similar to the anchor (as measured by the teacher’s embeddings) should exert a stronger influence on the loss, encouraging the student to learn finer-grained distinctions. Conversely, dissimilar negatives can be down-weighted since they are less informative.

To incorporate this structural information, we introduce a *semantic-weighted critic* in which the contribution of each negative pair is modulated by a teacher-defined weight, over negative samples in the same domain $w(T,S) : \mathcal{T} \times \mathcal{S} \rightarrow \mathbb{R}_{>0}$. We define it using teacher features:

$$
w(T,S)
=
1 + \alpha \exp \left( \frac{\cos(f_t(x_T), f_t(x_S))}{\tau} \right).
$$

Here $x_T$ and $x_S$ are the underlying samples that produced $T$ and $S$. Also $\alpha$ controls the strength of semantic modulation and $\tau$ is a temperature that adjusts sensitivity to similarity. This weight increases for teacher-near negative samples and smoothly decays for distant ones. The modified contrastive objective becomes

$$
\mathcal{L}_{\text{SW-CRD}}(h)
=
\mathbb{E}_{q(T,S|C=1)}[\log h(T,S)]
+
N \cdot \mathbb{E}_{q(T,S|C=0)}\big[
w(T, S) \cdot \log(1 - h(T,S))
\big].
$$

Although we introduce a semantic weighting term $w(T,S)$ to modulate the penalty on negative pairs, we prove in Appendix A that our objective $\mathcal{L}_{\text{SW-CRD}}$ remains a valid variational lower bound on the mutual information $I(T;S)$. Effectively, the weighting scheme shifts the implicit negative distribution to focus on 'hard' regions of the teacher's manifold, tightening the bound where discrimination is most difficult.

Intuitively, while standard CRD treats the negative manifold as a uniform background, our method effectively concentrates the probability density of negative samples that are semantically similar to the anchor. By introducing the weighting term $w(T,S)$, we shift the mutual information lower bound by a constant factor dependent on the teacher's geometry ($K = \mathbb{E}_{pos}[\log w]$). Maximizing this modified bound compels the student to assign high probability to positive pairs specifically relative to these weighted 'hard' negatives, thereby forcing the model to resolve fine-grained semantic distinctions that uniform sampling would ignore."

----




### 1. Formal Proposition (For Appendix)


**Proposition 1.** *Maximizing the semantic-weighted critic objective $\mathcal{L}_{\text{SW-CRD}}(h)$ with respect to the student encoder $f_S$ maximizes a lower bound on the mutual information $I(T;S)$ between teacher and student representations.*

**Proof.**
Let $p(T,S)$ denote the joint distribution and $p(T)p(S)$ the product of marginals. The objective is:
$$
\mathcal{L}_{\text{SW-CRD}}(h) = \mathbb{E}_{p(T,S)}[\log h(T,S)] + N \mathbb{E}_{p(T)p(S)}[w(T,S) \log(1 - h(T,S))].
$$
By pointwise maximization, the optimal critic $h^*$ for a fixed encoder is given by:
$$
h^*(T,S) = \frac{p(T,S)}{p(T,S) + N w(T,S) p(T)p(S)}.
$$
We relate this optimal critic to the mutual information. Considering $\log h^*(T,S)$:
$$
\log h^*(T,S) = \log \left( \frac{1}{1 + \frac{N w(T,S) p(T)p(S)}{p(T,S)}} \right) = - \log \left( 1 + \frac{N w(T,S) p(T)p(S)}{p(T,S)} \right).
$$
Using the inequality $-\log(1+x) \leq -\log(x)$ for $x>0$:
$$
\log h^*(T,S) \leq \log \left( \frac{p(T,S)}{p(T)p(S)} \right) - \log(N) - \log w(T,S).
$$
Taking the expectation over the joint distribution $p(T,S)$:
$$
\mathbb{E}_{p(T,S)}[\log h^*(T,S)] \leq \underbrace{\mathbb{E}_{p(T,S)} \left[ \log \frac{p(T,S)}{p(T)p(S)} \right]}_{I(T;S)} - \log N - \underbrace{\mathbb{E}_{p(T,S)}[\log w(T,S)]}_{K}.
$$
Rearranging for $I(T;S)$:
$$
I(T;S) \geq \mathbb{E}_{p(T,S)}[\log h^*] + \log N + K.
$$
Since $\log(1-h^*) < 0$ and $w(T,S) \geq 1$, the term $N \mathbb{E}_{p(T)p(S)}[w \log(1-h^*)]$ is non-positive. Adding it to the RHS maintains the lower bound:
$$
I(T;S) \geq \underbrace{\mathbb{E}_{p(T,S)}[\log h^*] + N \mathbb{E}_{p(T)p(S)}[w \log(1-h^*)]}_{\mathcal{L}_{\text{SW-CRD}}(h^*)} + \log N + K.
$$
Thus, $I(T;S) \geq \mathcal{L}_{\text{SW-CRD}} + C$, where $C$ is a constant independent of the student encoder. $\hfill \square$



----






### Part 2: Formal Derivation for Appendix

Here is a rigorous derivation formatted for an academic appendix. It moves from the integral definition to the pointwise solution.

#### **Appendix B: Derivation of the Optimal Semantic-Weighted Critic**

**Proposition.** *The optimal critic function $h^*(T,S)$ that maximizes the semantic-weighted objective $\mathcal{L}_{\text{SW-CRD}}$ is given by:*
$$
h^*(T,S) = \frac{p(T,S)}{p(T,S) + N \cdot w(T,S) \cdot p(T)p(S)}
$$

**Proof.**
The semantic-weighted objective function is defined as:
$$
\mathcal{L}(h) = \mathbb{E}_{p(T,S)}[\log h(T,S)] + N \cdot \mathbb{E}_{p(T)p(S)}\big[ w(T,S) \cdot \log(1 - h(T,S)) \big]
$$

We can express the expectations as integrals over the domains of $T$ and $S$:
$$
\mathcal{L}(h) = \iint \Big( p(T,S) \log h(T,S) + N \cdot w(T,S) p(T)p(S) \log(1 - h(T,S)) \Big) \, dT \, dS
$$

To maximize the functional $\mathcal{L}(h)$ with respect to the function $h$, it is sufficient to maximize the integrand pointwise for every pair $(T,S)$. Let $y = h(T,S)$ be the value of the critic at a specific pair. We define the scalar objective function $J(y)$ for this fixed pair as:
$$
J(y) = A \cdot \log(y) + B \cdot \log(1 - y)
$$
where the coefficients $A$ and $B$ are non-negative constants specific to the pair $(T,S)$:
$$
A = p(T,S)
$$
$$
B = N \cdot w(T,S) \cdot p(T)p(S)
$$

To find the optimal value $y^*$, we take the derivative of $J(y)$ with respect to $y$ and set it to zero:
$$
\frac{dJ}{dy} = \frac{A}{y} - \frac{B}{1 - y} = 0
$$

Solving for $y$:
$$
\begin{aligned}
\frac{A}{y} &= \frac{B}{1 - y} \\
A(1 - y) &= B y \\
A - Ay &= B y \\
A &= y(A + B) \\
y &= \frac{A}{A + B}
\end{aligned}
$$

We substitute the definitions of $A$ and $B$ back into the expression for $y$:
$$
h^*(T,S) = \frac{p(T,S)}{p(T,S) + N \cdot w(T,S) \cdot p(T)p(S)}
$$

To confirm this is a maximum, we observe the second derivative:
$$
\frac{d^2J}{dy^2} = -\frac{A}{y^2} - \frac{B}{(1-y)^2}
$$
Since $A, B \ge 0$ and $y \in (0,1)$, the second derivative is strictly negative, confirming that $h^*(T,S)$ is the global maximum. $\hfill \square$












-------------------------------












## 0. Setup (same as CRD, plus (w))

We keep exactly the same probabilistic setup as in your derivation:

* Positive (joint) pairs:
  $$
  q(T,S\mid C=1) = p(T,S).
  $$
* Negative (product of marginals) pairs:
  $$
  q(T,S\mid C=0) = p(T)p(S).
  $$
* Priors (1 positive : (N) negatives):
  $$
  q(C=1) = \frac{1}{N+1},\quad q(C=0) = \frac{N}{N+1}.
  $$

Your **weighted critic objective** is
$$
\mathcal{L}_{\text{mod}}(h)
=
\mathbb{E}_{q(T,S\mid C=1)}[\log h(T,S)]
+
N \cdot \mathbb{E}_{q(T,S\mid C=0)}\big[
w(T,S) \cdot \log(1 - h(T,S))
\big].
$$
with
$$
w(T,S) = 1 + \alpha \exp\big(\cos(f_t(i), f_t(j))/\tau\big) \ge 1.
$$

So (w(T,S)>0) and independent of the **student** and the critic parameters (it depends only on the teacher geometry and hyper-parameters).

---

## 1. Optimal critic under the weighted loss

We first find the optimal critic $h_w^*$ under this new objective.

Write
$$
p_+(T,S) := p(T,S), \qquad p_-(T,S) := p(T)p(S).
$$

Then
$$
\mathcal{L}_{\text{mod}}(h)
=

\int p_+(T,S)\log h(T,S)dT dS
+
N\int p_-(T,S)w(T,S)\log(1-h(T,S))dT dS.
$$

This is a pointwise functional optimization. For each $(T,S)$, define
$$
a := p_+(T,S),\quad b := N p_-(T,S) w(T,S),
$$
and consider the scalar function
$$
\ell(h) = a\log h + b \log(1-h),\quad h\in(0,1).
$$

Differentiating:
$$
\frac{d\ell}{dh} = \frac{a}{h} - \frac{b}{1-h}.
$$
Setting to zero gives
$$
\frac{a}{h} = \frac{b}{1-h}
\quad\Rightarrow\quad
h_w^*(T,S)
=

\frac{a}{a+b}

\frac{p(T,S)}{p(T,S) + N w(T,S) p(T)p(S)}.
$$

So the **optimal critic under the weighted loss** is
$$
h_w^*(T,S)
=

\frac{p(T,S)}{p(T,S) + N w(T,S) p(T)p(S)}.
\tag{★}
$$

This is exactly the same form as the unweighted CRD posterior, but with $p(T)p(S)$ replaced by $w(T,S)p(T)p(S)$.

---

## 2. Interpreting $h_w^*$ as a posterior

Define an *effective* generative model over $(T,S,C)$ with:

* Positive conditional unchanged:
  $$
  q_w(T,S\mid C=1) = p(T,S).
  $$
* Weighted negative conditional:
  $$
  \tilde q_w(T,S\mid C=0)
  \propto w(T,S)p(T)p(S).
  $$

Concretely, let
$$
\tilde q_w(T,S\mid C=0)
=
\frac{w(T,S)p(T)p(S)}{Z},
\quad
Z = \int w(T,S)p(T)p(S)dT dS.
$$

Keep the same priors $q_w(C=1)=\frac{1}{N+1}, q_w(C=0)=\frac{N}{N+1}$.

Then Bayes’ rule gives the posterior
$$
q_w(C=1\mid T,S)
=

\frac{p(T,S) \frac{1}{N+1}}{p(T,S) \frac{1}{N+1} + \tilde q_w(T,S\mid C=0) \frac{N}{N+1}}.
$$

Plug in $\tilde q_w$:
$$
q_w(C=1\mid T,S)
=
\frac{p(T,S)}{p(T,S) + N\tilde q_w(T,S\mid C=0)}.
$$

Up to the normalizing constant $Z$, $\tilde q_w(T,S\mid C=0) \propto w(T,S)p(T)p(S)$, so **up to a constant absorbed into (N)** this posterior has the same functional form as $(★)$. Equivalently, you can think of using the same $p(T,S)$ but a *reweighted* negative density.

So: maximizing $\mathcal{L}_{\text{mod}}(h)$ is equivalent to learning the Bayes-optimal classifier between (C=1) and (C=0) under a mixture model where negatives are drawn from a weighted version of (p(T)p(S)).

---

## 3. MI lower bound with semantic weights

Now we want a bound of the form “MI ≥ something involving $\log h_w^*$”.

Start from the exact expression for the posterior derived from $(★)$ but using the *true* product-of-marginals $p(T)p(S)$ (no normalization):

$$
q_w(C=1\mid T,S)
=
\frac{p(T,S)}{p(T,S) + N w(T,S) p(T)p(S)}.
$$

Take logs and manipulate as in the original CRD derivation:

$$
\log q_w(C=1\mid T,S)
=
- \log\left(1 + \frac{N w(T,S) p(T)p(S)}{p(T,S)}\right).
$$

Let
$$
x = \frac{N w(T,S) p(T)p(S)}{p(T,S)} > 0.
$$

Using the same inequality CRD uses:
$$
-\log(1+x) \le -\log x
\quad\text{for } x>0,
$$
we get

$$
\log q_w(C=1\mid T,S)
\le
- \log\left(\frac{N,w(T,S),p(T)p(S)}{p(T,S)}\right)
  =
  -\log N - \log w(T,S) + \log\frac{p(T,S)}{p(T)p(S)}.
$$

Take expectation w.r.t. $q(T,S\mid C=1)=p(T,S)$:

$$
\mathbb{E}_{p(T,S)}[\log q_w(C=1\mid T,S)]
\le
-\log N - \mathbb{E}_{p(T,S)}[\log w(T,S)]
+
\mathbb{E}_{p(T,S)}\Big[\log\frac{p(T,S)}{p(T)p(S)}\Big].
$$

Recognize the MI term:
$$
I(T;S) = \mathbb{E}_{p(T,S)}\Big[\log\frac{p(T,S)}{p(T)p(S)}\Big],
$$
so rearranging:

$$
I(T;S)
\ge
\log N
+
\mathbb{E}_{p(T,S)}[\log q_w(C=1\mid T,S)]
+
\mathbb{E}_{p(T,S)}[\log w(T,S)].
\tag{1}
$$

Crucially:

* $q_w(C=1\mid T,S)$ is **exactly** the optimal critic ($h_w^*(T,S)$).
* $w(T,S)$ depends **only** on the teacher and fixed hyper-parameters ($( \alpha,\tau )$), not on the student or critic parameters.

So we already have an MI lower bound of the form

$$
I(T;S)
\ge
\log N
+
\mathbb{E}_{q(T,S\mid C=1)}[\log h_w^*(T,S)]
+
\mathbb{E}_{q(T,S\mid C=1)}[\log w(T,S)  ].
$$

---

## 4. Weakening step with weighted negatives

Exactly as in the original CRD paper, we now weaken the bound by **adding** a negative term that is always $\le 0$.

Since $h_w^*(T,S)\in(0,1)$ and $w(T,S)\ge 0$,

$$
\log(1 - h_w^*(T,S)) \le 0
\quad\Rightarrow\quad
w(T,S),\log(1 - h_w^*(T,S)) \le 0.
$$

Therefore, with $N>0$,

$$
N \mathbb{E}_{q(T,S\mid C=0)}[w(T,S) \log(1 - h_w^*(T,S))] \le 0.
$$

From inequality (1),

$$
I(T;S)
\ge
\log N
+
\mathbb{E}_{C=1}[\log h_w^*]
+
\mathbb{E}_{C=1}[\log w].
$$

Adding the non-positive term on the RHS **keeps the inequality valid but looser**:

$$
I(T;S)
\ge
\log N
+
\underbrace{\Big(
\mathbb{E}_{C=1}[\log h_w^*]
+
N \mathbb{E}_{C=0}[w\log(1 - h_w^*)]
\Big)}_{\mathcal{L}_{\text{mod}}(h_w^*)}
+
\mathbb{E}_{C=1}[\log w].
$$

So we obtain

$$
I(T;S)
\ge
\log N
+
\mathcal{L}_{\text{mod}}(h_w^*)
+
\mathbb{E}_{q(T,S\mid C=1)}[\log w(T,S)].
\tag{2}
$$
Define the constant
$$
K := \mathbb{E}_{q(T,S\mid C=1)}[\log w(T,S)],
$$
which depends only on the teacher/hyper-parameters, not on the student encoder $f^S$ or critic parameters.

Then (2) becomes

$$
I(T;S)
\ge
\log N + K + \max_h \mathcal{L}_{\text{mod}}(h).
$$

Thus, **maximizing $\mathcal{L}_{\text{mod}}(h)$ over $(h, f^S)$ still maximizes a valid lower bound on the mutual information $I(T;S)$**, up to the additive constant $(\log N + K)$.

---

## 5. Takeaways (what this means for your method)

1. The semantic weighting simply changes the Bayes-optimal critic from
   $$
   h^*(T,S) = \frac{p(T,S)}{p(T,S) + N p(T)p(S)}
   $$
   to
   $$
   h_w^*(T,S) = \frac{p(T,S)}{p(T,S) + N w(T,S) p(T)p(S)}.
   $$

2. Using the same MI-bound trick as CRD, we get
   $$
   I(T;S)
   \ge
   \log N + \mathbb{E}_{C=1}[\log h_w^*] + \mathbb{E}_{C=1}[\log w],
   $$
   so the **MI bound is preserved**, with an extra constant shift $(\mathbb{E}[\log w])$.

3. After the usual “weakening” step (adding the negative term), your weighted critic objective $(\mathcal{L}_{\text{mod}}(h))$ is still directly tied to that MI lower bound:
   $$
   I(T;S)
   \ge
   \log N + K + \max_h \mathcal{L}_{\text{mod}}(h),
   $$
   where $(K)$ is constant w.r.t. student and critic parameters.

So mathematically: **yes, your semantic-weighted objective is still optimizing an MI lower bound in exactly the same variational sense as CRD**, just with a modified (teacher-aware) posterior and an additive constant in the bound.