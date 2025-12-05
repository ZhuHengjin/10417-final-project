Here is the full, step-by-step derivation of the Contrastive Representation Distillation (CRD) loss, tracing the path from the posterior probability $q(C=1|T,S)$ to the Mutual Information (MI) bound, and finally to the computable critic loss.

This derivation expands on the steps found in **Section 3.1**  and **Appendix 6.2.1**  of the provided paper.

### 1. The Probabilistic Setup
We define a classification task to distinguish between "positive" pairs (drawn from the joint distribution) and "negative" pairs (drawn from the product of marginals).

**Definitions:**
* $T = f^T(x)$: Teacher representation.
* $S = f^S(x)$: Student representation.
* $C$: A latent variable (class label).
    * $C=1$: The pair $(T, S)$ is drawn from the joint distribution $p(T, S)$ (i.e., from the same input $x$).
    * $C=0$: The pair $(T, S)$ is drawn from the product of marginals $p(T)p(S)$ (i.e., different inputs).
* $N$: The number of negative samples for every 1 positive sample.

**Priors:**
Because there is 1 positive sample for every $N$ negative samples, the prior probabilities are:
$$q(C=1) = \frac{1}{N+1}, \quad q(C=0) = \frac{N}{N+1}$$

**Likelihoods:**
$$q(T,S|C=1) = p(T,S)$$
$$q(T,S|C=0) = p(T)p(S)$$

### 2. Deriving the Posterior $q(C=1|T,S)$
We apply Bayes' Theorem to find the probability that a given pair $(T, S)$ comes from the joint distribution (is a positive pair).

$$
q(C=1|T,S) = \frac{q(T,S|C=1)q(C=1)}{q(T,S|C=1)q(C=1) + q(T,S|C=0)q(C=0)}
$$

Substitute the priors and likelihoods defined above:

$$
q(C=1|T,S) = \frac{p(T,S) \cdot \frac{1}{N+1}}{p(T,S) \cdot \frac{1}{N+1} + p(T)p(S) \cdot \frac{N}{N+1}}
$$

Multiply the numerator and denominator by $(N+1)$ to clear the fractions:

$$
q(C=1|T,S) = \frac{p(T,S)}{p(T,S) + N p(T)p(S)} \quad \text{(Eq. 7)}
$$

### 3. Deriving the Mutual Information (MI) Bound
The goal is to maximize the Mutual Information $I(T;S)$. Recall the definition of MI:
$$I(T;S) = \mathbb{E}_{p(T,S)} \left[ \log \frac{p(T,S)}{p(T)p(S)} \right]$$

To link the posterior (Eq. 7) to MI, we take the log of the posterior and perform algebraic manipulation.

**Step 3a: Log-Posterior Manipulation**
$$
\log q(C=1|T,S) = \log \left( \frac{p(T,S)}{p(T,S) + N p(T)p(S)} \right)
$$

Invert the fraction inside the log (which negates the log):
$$
\log q(C=1|T,S) = - \log \left( \frac{p(T,S) + N p(T)p(S)}{p(T,S)} \right)
$$
$$
\log q(C=1|T,S) = - \log \left( 1 + \frac{N p(T)p(S)}{p(T,S)} \right)
$$

**Step 3b: Applying the Inequality**
We use the inequality $-\log(1+x) \leq -\log(x)$ (since $1+x > x$ for positive $x$, and $\log$ is monotonic).
$$
\log q(C=1|T,S) \leq - \log \left( \frac{N p(T)p(S)}{p(T,S)} \right)
$$
$$
\log q(C=1|T,S) \leq - \log(N) - \log \left( \frac{p(T)p(S)}{p(T,S)} \right)
$$
$$
\log q(C=1|T,S) \leq - \log(N) + \log \left( \frac{p(T,S)}{p(T)p(S)} \right) \quad \text{(Eq. 8)}
$$

**Step 3c: Taking the Expectation**
Take the expectation of both sides with respect to the joint distribution $q(T,S|C=1) = p(T,S)$:
$$
\mathbb{E}_{q(T,S|C=1)} [\log q(C=1|T,S)] \leq -\log(N) + \underbrace{\mathbb{E}_{p(T,S)} \left[ \log \frac{p(T,S)}{p(T)p(S)} \right]}_{I(T;S)}
$$

Rearranging to isolate $I(T;S)$, we get the **MI lower bound**:
$$
I(T;S) \geq \log(N) + \mathbb{E}_{q(T,S|C=1)} [\log q(C=1|T,S)] \quad \text{(Eq. 9)}
$$

### 4. Introducing the Critic $h$ and the Loss Function
Since the true distributions $p(T,S)$ and $p(T)p(S)$ are unknown, we cannot compute $q(C=1|T,S)$ directly. We approximate it using a model $h(T,S)$ (the critic) where $h: \{\mathcal{T}, \mathcal{S}\} \rightarrow [0, 1]$.

We train $h$ to distinguish between samples from the joint (Class 1) and marginals (Class 0) by maximizing the log-likelihood (binary cross-entropy):

$$
\mathcal{L}_{critic}(h) = \mathbb{E}_{q(T,S|C=1)}[\log h(T,S)] + N \mathbb{E}_{q(T,S|C=0)}[\log(1 - h(T,S))] \quad \text{(Eq. 10)}
$$

*Note: The factor $N$ appears because there are $N$ times more negative samples than positive samples in the data distribution.*

**Why substitute $h$ into the MI bound?**
According to the Gibbs inequality (proven in Appendix 6.2.1), the optimal critic $h^*$ that maximizes $\mathcal{L}_{critic}$ is exactly the true posterior:
$$h^*(T,S) = q(C=1|T,S)$$
Thus, we can substitute $h^*$ into Eq. 9.

### 5. Deriving the Final Computable Objective
Substituting $h^*$ into Eq. 9 gives a valid bound:
$$I(T;S) \geq \log(N) + \mathbb{E}_{q(T,S|C=1)} [\log h^*(T,S)] \quad \text{(Eq. 12)}$$

However, optimizing this directly is difficult because $h^*$ depends on the student representation. The authors apply a "weakening" step to create a computable loss function that allows joint optimization.

**The Weakening Step:**
We add a term to the Right Hand Side (RHS). The term is $N \mathbb{E}_{q(T,S|C=0)} [\log(1 - h^*(T,S))]$.
Since probabilities are $\leq 1$, the log term is negative. Since $N > 0$, the entire term is **negative (or zero)**.
Adding a negative number to a lower bound keeps the inequality valid (it just makes the bound looser).

$$
\begin{aligned}
I(T;S) &\geq \log(N) + \mathbb{E}_{q(T,S|C=1)} [\log h^*] \\
&\geq \log(N) + \underbrace{\mathbb{E}_{q(T,S|C=1)} [\log h^*] + N \mathbb{E}_{q(T,S|C=0)} [\log(1 - h^*)]}_{\text{This is exactly } \mathcal{L}_{critic}(h^*)}
\end{aligned}
$$

Thus:
$$I(T;S) \geq \log(N) + \max_h \mathcal{L}_{critic}(h)$$

### 6. Final Loss Formulation
To learn the student representation $f^S$ that captures the most information from the teacher, we maximize this lower bound:

$$f^{S*} = \arg \max_{f^S} \max_h \mathcal{L}_{critic}(h)$$

In practice, this maximization is converted to a minimization of the negative loss. Using the specific parameterized form of $h$ (Eq. 19), the final loss minimizes:

$$
\mathcal{L}_{CRD} = - \left( \mathbb{E}_{C=1}[\log h(T,S)] + N \mathbb{E}_{C=0}[\log (1 - h(T,S))] \right)
$$