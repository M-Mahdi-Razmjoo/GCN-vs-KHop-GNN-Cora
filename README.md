# GCN vs. k-Hop GNN on Cora Dataset

This project explores Graph Neural Networks (GNNs), focusing specifically on the Graph Convolutional Network (GCN) model. We use the well-known **Cora** dataset to evaluate two different GCN architectures:

- A standard stacked GCN with multiple 1-hop message-passing layers.
- A custom GCN variant that performs k-hop message aggregation in a single layer.

## Objectives

- Visualize high-dimensional node features in 2D using **t-SNE**, based on node labels.
- Implement a custom `GCNConv` class from scratch for greater control over the message-passing process.
- Investigate the effect of aggregating messages from neighbors at distance up to `k` in a single layer.
- Compare the performance of:
  - `StackedGCN`: Traditional multi-layer GCN using 1-hop message passing.
  - `SingleKHopGCN`: Single-layer GCN with extended message aggregation up to k hops.
- Evaluate the models on the Cora dataset.
- Optionally, use an MLP layer after GCN layers to improve classification accuracy.

## Dataset

We use the [Cora dataset](https://linqs.soe.ucsc.edu/data), a citation network where nodes represent papers and edges represent citations. Each node is classified into one of several research topics.

## Model Architectures

### 1. StackedGCN

- Multiple GCN layers.
- Each layer aggregates information from immediate (1-hop) neighbors.

### 2. SingleKHopGCN

- A single custom GCN layer.
- Aggregates features from all nodes up to `k` hops away in one pass.
- Implemented from scratch to allow for k-hop aggregation.

> Note: When `k = 1`, the behavior is identical to the standard GCN.

## Dimensionality Reduction

To visualize the high-dimensional input features, we apply **t-SNE** for 2D projection. This helps illustrate how well the model learns to separate different classes in the latent space.

