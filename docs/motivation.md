# Motivation: Semi-Supervised Node Classification with Spectral Feature Augmentation

## The problem

On citation networks like Cora (2,708 nodes, 7 classes), only ~5% of nodes have labels — the semi-supervised node classification setting. Training a GNN classifier from 140 labeled examples is data-starved. Performance plateaus, and the model is vulnerable to distributional quirks of the training subset.

Data augmentation is the standard response. Existing techniques inject generic noise into features:

- **DropFeature:** randomly zero feature entries
- **FeatMask:** mask whole feature dimensions
- **GraphMix / mixup-style:** linear interpolation between node feature vectors

These treat features as unstructured vectors. They ignore that features on a graph have spectral structure aligned with the graph's community organization — the very structure GNNs actually exploit for classification.

## The proposed solution

**Spectral feature augmentation:** generate augmented training features by adding noise in the Laplacian eigenbasis proportional to per-mode energy. Each augmented sample preserves the spectral energy profile of the original features by construction. Downstream GNNs see augmented inputs that look like plausible draws from the same spectral distribution as the real features.

This is directly motivated by the diffusion-model finding in Phase 6b: on Cora, spectral augmentation produces generated features with 46% lower spectral Wasserstein distance to real features than gaussian augmentation. Spectral structure is what makes augmented features "realistic" from the perspective of any GNN downstream.

## Why this matters

**Standard GNNs learn via polynomial filters on the graph Laplacian.** The eigenvectors of L are the basis in which GNN filters operate. If augmented features distort the spectral profile (the variance distribution across eigenmodes), the GNN sees unrealistic inputs and learns spurious patterns.

Spectral augmentation is the minimal principled extension of existing augmentation: instead of distorting structure, it adds variation that respects it. The augmented features serve as "plausible new node contents" that preserve the properties the GNN actually uses.

## Applications

### Primary: Label-scarce node classification

The most direct application. For each benchmark graph (Cora, CiteSeer, PubMed, Amazon Photo, Coauthor CS):

1. Train a standard GCN classifier using only the labeled subset's features.
2. Compare classifier accuracy with: no augmentation / gaussian / dropout / spectral augmentation of training features.
3. Hypothesis: spectral augmentation yields higher test accuracy than alternatives, particularly when the label fraction is small (< 5%).

This is directly testable, connects to standard benchmarks, and the improvement is measurable in accuracy — not just in distributional metrics like W1.

### Secondary: Privacy-preserving feature release

Many organizations have graph-structured data with sensitive features (healthcare: patient-contact graphs with clinical attributes; finance: transaction networks with account data; social platforms: user profiles). They cannot release real feature matrices, but would like to enable downstream research.

A spectral diffusion model trained on real features generates synthetic feature matrices that preserve spectral structure (so downstream GNN analyses give similar results) without exposing specific records. Differential-privacy extensions strengthen this: bounded-DP training guarantees on the diffusion model propagate to the released synthetic dataset.

### Tertiary: Feature imputation under systematic missingness

Real networks have missing features in a non-random way — new users without profiles, inactive accounts, uncharacterized proteins. Standard imputation (mean, neighbor averaging) ignores global spectral structure.

Diffusion conditioned on observed features + graph produces imputed features consistent with the spectral profile of the observed part. This is the natural inverse problem: given partial observations on a graph, complete them consistently.

### Additional: Anomaly detection via density estimation

A trained diffusion model gives p(x | G). Nodes whose features have low likelihood under the model are anomalous. Applies to fraud detection on payment networks, bot detection on social networks, intrusion detection on communication graphs. The spectral structure makes this more discriminative than topology-free density estimation.

## Why spectral augmentation specifically, not just any augmentation

The 46% W1 improvement over gaussian augmentation (Phase 6b on Cora) is the empirical starting point. But the underlying reason is mechanical:

- Gaussian noise has flat spectral variance — it corrupts low-frequency (community-encoding) modes as much as high-frequency (local-detail) modes.
- Spectral augmentation allocates noise proportional to the mode's current energy — preserving the shape of the energy profile the GNN relies on.

For a label-scarce classifier trying to exploit community structure, feature augmentations that destroy community-scale structure are counterproductive. Spectral augmentation avoids this failure mode by construction.

## What still needs to be shown

The current evidence is one dataset (Cora), one metric (spectral W1), no downstream task. To make this story publishable:

1. **Cross-dataset validation** (Phase 7a): spectral > gaussian augmentation on Cora, PubMed, Amazon Photo, Coauthor CS on spectral W1.
2. **Downstream utility** (Phase 7b): spectral augmentation of training features improves semi-supervised node classification accuracy over alternatives. Tested at multiple label fractions (1%, 5%, 10%) on multiple datasets.
3. **Baselines and ablations:** comparison to DropFeature, FeatMask, GraphMix, FLAG, and other standard graph-augmentation techniques. Ablation on sigma hyperparameter.

Phase 2g's pattern — 12.5% W1 improvement giving only 0.9% classification accuracy gain — is the risk to beat. If spectral augmentation gives +3-5% accuracy improvement over best existing augmentation in label-scarce settings, the paper has a concrete and defensible claim. If the accuracy gain is <1%, the contribution becomes purely methodological.

## Scope

This paper addresses: how to augment features on a fixed graph when training data is limited.

This paper does not address:
- Generation of new graph topologies (that's molecular-style graph generation)
- Training on multiple graphs simultaneously (that's ensemble graph learning)
- Augmentation of the graph structure itself (that's a separate line: GraphCL, GRAND)

By working in a single-graph regime and treating features as the variable to augment, we get a clean experimental setup that matches the practical semi-supervised learning use case and directly addresses a well-motivated problem.
