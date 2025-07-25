# Reproduction

Install crosshair

    uvx crosshair-tool
    
Execute `./cluster.py se` to produce clusters using symbolic execution.

Execute `./cluster.py test` to produce clusters using tests.

Execute `./cluster.py diff` to measure average Rand index.


# Adjusted Rand Index (ARI)

- The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings
- Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
