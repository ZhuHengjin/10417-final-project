

## Knowledge Distillation (KD)


Knowledge Distillation (Hinton et al., 2015) sets the core framework for transferring knowledge from a large teacher model $T$ to a small student model $S$. The total loss in KD is defined as a weighted combination of the cross-entropy loss and the KL divergence between the output distributions of the teacher and student:

$$
\mathcal{L}_{\text{KD}} = (1 - \alpha)\, \mathcal{L}_{\text{CE}}(y, p_S) + \alpha T^2 \, \text{KL}(p_T \| p_S)
$$

where $\alpha$ is a hyperparameter that balances the two loss terms, and $p_T$ and $p_S$ denote teacher and student outputs. In our preliminary experiment, we use $\alpha = 0.9$.

## Relational Knowledge Distillation (RKD)

Relational Knowledge Distillation (RKD) (Park et al., 2019) transfers relational information between pairwise data points rather than absolute features. It matches the pairwise distances and angles among embeddings:

$$
\mathcal{L}_{\text{RKD}} = \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{angle}}
$$

In our experiment, the total loss is a sum of $\mathcal{L}_{CE} + \mathcal{L}_{RKD}$.

## Contrastive Representation Distillation (CRD)

This serves as our primary baseline, providing a foundation upon which our proposed semantic-aware approach will be built. Instead of matching logits, Contrastive Representation Distillation (CRD) (Tian et al., 2022) encourages alignment between the student and teacher feature embeddings using a contrastive objective:

$$
\mathcal{L}_{\text{CRD}} = - \log \frac{\exp(f_t^\top f_s / T)}{\sum_{i=1}^{N} \exp(f_t^\top f_{s,i} / T)}
$$

where $f_t$ and $f_s$ are the teacher and student features for a positive pair, $f_{s,i}$ are negative samples from other instances, and $T$ is a temperature parameter. In our experiment, the total loss is a weighted sum of $\mathcal{L}_{CE} + 0.8\mathcal{L}_{CRD}$.

By aligning intermediate representations through a contrastive objective, it encourages the student to learn embeddings that preserve the structure of the teacher's output. This property makes CRD an ideal starting point for extending toward semantic-aware relational contrastive learning, as it already emphasizes the relationships between samples, which can then be further refined by semantic weighting.

## Implementation and Results

All baseline models were implemented using the [RepDistiller framework](https://github.com/HobbitLong/RepDistiller), which provides reference implementations for various knowledge distillation methods. We used the released codebase as a foundation and adapted it to our experimental setting. 

In our preliminary experiment, we used a **ResNet-110 teacher** and trained **ResNet-32 students** on the **CIFAR-100 dataset** for 240 epochs.

## Improved SRCD loss

$$
\mathcal{L}_{\text{SRCD}} = \mathcal{L}_{\text{CRD}} + \lambda \mathcal{L}_{\text{semantic}}
$$

### Semantic-Aware Relational Weighting

$$
w_{ij} = \frac{\exp (\cos (s_i, s_j) / \tau)}{\sum_{k=1}^{N} \exp (\cos (s_i, s_k) / \tau)}
$$

or define it as:

$$
w_{ij} = \mathbf{1}(y_i = y_j) \cdot \frac{\exp(\cos(f_t(i), f_t(j))/\tau)}{\sum_{k:y_k=y_i} \exp(\cos(f_t(i), f_t(k))/\tau)}
$$

* **$\mathbf{1}(y_i = y_j)$**: An indicator function that ensures the weight is non-zero only if samples $i$ and $j$ belong to the same class (positive pairs).
* **$\cos(f_t(i), f_t(j))$**: The cosine similarity between the teacher's feature representations for samples $i$ and $j$.
* **$\tau$**: A temperature hyperparameter scaling the distribution.

### Weighted relational distance

Define:

$$
d_t(i, j) = \|f_t(i) - f_t(j)\|_2
$$

$$
d_s(i, j) = \|f_s(i) - f_s(j)\|_2
$$

Then:

$$
L_{\text{semantic-dist}} = \sum_{i,j} w_{ij} (d_s(i, j) - d_t(i, j))^2
$$

### Weighted relational angle
$$
\theta_t(i, j, k) = \angle(f_t(i) - f_t(k), f_t(j) - f_t(k))
$$

$$
L_{\text{semantic-angle}} = \sum_{i,j,k} w_{ij}w_{ik} (\theta_s(i, j, k) - \theta_t(i, j, k))^2
$$

### Combined semantic loss

$$
L_{\text{semantic-rel}} = \beta_1 L_{\text{semantic-dist}} + \beta_2 L_{\text{semantic-angle}}
$$

### Impoved sematic CRD

$$
\mathcal{L}_{\text{CRD}} 
= - \log \frac{\exp(f_t^\top f_s / T)}
{\sum_{i=1}^{N} w_{i} \cdot \exp(f_t^\top f_{s,i} / T)}
$$

### Final improved CRD (SRCD) loss

Put everything together:

$$
L_{\text{SRCD}} = L_{\text{CRD}} + \lambda \Bigl[ \beta_1 \sum_{i,j} w_{ij}(d_s - d_t)^2 + \beta_2 \sum_{i,j,k} w_{ij}w_{ik}(\theta_s - \theta_t)^2 \Bigr]
$$

Where:
* $L_{\text{CRD}}$: contrastive InfoNCE alignment
* $w_{ij}$: semantic importance of pair
* $d_s, d_t$: pairwise distances
* $\theta_s, \theta_t$: relational angles
* $\lambda, \beta_1, \beta_2$: hyperparameters

Your intuition to add semantic awareness is excellent. To further boost performance, you can add orthogonal "modules" that address other blind spots in the standard RKD/CRD framework.

Based on recent advancements (like DCD and GSKD), here are three concrete improvements you can add to your loss function:

## Adaptive Temperature (The "Difficulty" Awareness)
Standard distillation uses a fixed temperature ($T$) for all samples. However, some samples are "easy" (teacher is very confident) and others are "hard" (teacher is ambiguous).
* **The Problem:** A fixed $T$ treats clear images and confusing images the same way.
* **The Fix:** Make $T$ dynamic per sample. Harder samples (high entropy in teacher output) often contain more "dark knowledge" and benefit from a higher temperature to smooth the distribution, while easy samples can use a lower temperature.
* **Implementation:**
    Modify your loss to scale $T$ based on the teacher's confidence (max logit or entropy).
    $$T_i = T_{base} \times (1 + \alpha \cdot \mathbb{H}(P_t(x_i)))$$
    Where $\mathbb{H}$ is entropy. This forces the student to "pay more attention" to the nuances of difficult samples.

## Consistency Regularization (The "Self" Awareness)
Right now, your student only looks at the **Teacher**. It should also look at **Itself** to ensure internal stability.
* **The Problem:** The student might mimic the teacher correctly for an image $x$, but give a totally different embedding for a slightly rotated version of $x$ (image $x'$).
* **The Fix:** Enforce that the student's representation remains stable across different augmentations of the same image.
* **Implementation:**
    Add a consistency term that ignores the teacher and focuses on self-robustness.
    $$\mathcal{L}_{\text{cons}} = \| f_s(x) - f_s(\text{aug}(x)) \|^2$$
    *Note: This is "free" supervision because it doesn't require the teacher's forward pass, just a second pass of the student.*

<!-- ### 3. Global Structural Similarity (The "Batch" Awareness)
RKD looks at *pairs* (distances/angles). You can upgrade this to look at the entire *batch topology* at once using a similarity matrix (often called "Attention Transfer" or "Gram Matrix matching").
* **The Problem:** RKD checks if "A is close to B". It often misses the global context of "How does A relate to the entire cluster of B, C, D, and E combined?"
* **The Fix:** Compute the self-similarity matrix for the entire batch and force the student's matrix to match the teacher's matrix.
* **Implementation:**
    Let $G_t$ be the $N \times N$ matrix where $G_{ij} = \frac{f_{ti} \cdot f_{tj}}{\|f_{ti}\| \|f_{tj}\|}$ (Teacher's view of the batch geometry).
    Minimize the Frobenius norm between the Student's map ($G_s$) and Teacher's map ($G_t$):
    $$\mathcal{L}_{\text{global}} = \| G_t - G_s \|_F^2$$ -->

