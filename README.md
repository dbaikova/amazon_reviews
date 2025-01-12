# Amazon Reviews Recommender Systems

This project develops and evaluates multiple recommendation algorithms using Amazon review data.

## Approaches

- **Content-Based Filtering**
  - Identify similar items based on text features. Use LSA to capture latent relationships.

- **Collaborative Filtering**
  - Implement a user-based recommendation system. Calculate user-user similarity using cosine similarity on a user-item matrix and ranks items based on predicted scores.

- **Hybrid Approaches**: Combine user and item features for recommendations
  - **Gradient Boosting**: Train a CatBoostRanker with PairLogit loss for ranking.
  - **Neural Networks** Build a Neural Collaborative Filtering (NCF)-based hybrid recommender using pairwise ranking.

## Dataset

This project utilizes the dataset from the following study:

> **Hou, Yupeng, et al.** "Bridging Language and Items for Retrieval and Recommendation." *arXiv preprint arXiv:2403.03952* (2024).  
> [https://arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952)
