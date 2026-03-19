# Neuralized KMeans with LRP for Interpretable Text Clustering

## 📌 Overview

This project implements an interpretable clustering method based on:

- KMeans clustering
- Neuralization of KMeans into a neural network
- Layer-wise Relevance Propagation (LRP)

The goal is to explain **why each document is assigned to a cluster**, and trace this decision back to:

- embedding features
- individual words (tokens)

---

## 🧠 Full Pipeline (What this project does)

The complete process is:

### 1. Text preprocessing
- Lowercasing
- Removing symbols
- Stopword removal
- Tokenization

---

### 2. Word embeddings
Each word is converted into a vector using:

- Word2Vec (Google News 300-dimensional embeddings)

---

### 3. Document embeddings
Each document is represented as:

- the **mean of its word embeddings**

---

### 4. Clustering
- KMeans is applied to document embeddings
- Output:
  - cluster labels
  - cluster centroids

---

### 5. Neuralization of KMeans
KMeans is converted into a neural network:

- Linear layer:
  
  h_k = w_k · x + b_k

- where weights and bias come from centroids

---

### 6. Relevance Propagation (LRP)

Relevance is propagated backward:

#### Step 1 — Hidden layer relevance
- distributes importance across clusters

#### Step 2 — Input feature relevance
- assigns importance to each embedding dimension

---

### 7. Token-level relevance
Feature relevance is distributed back to words:

- each word gets a relevance score
- shows which words influenced the cluster assignment

## 📚 Datasets

The method was tested on:

- AG News Dataset (4 categories)
- BBC News Dataset (5 categories)
- E-commerce Text Dataset

⚠️ Datasets are not included due to licensing restrictions.

Please download them from their original sources.

---

## 🧠 What the Code Does

The file `lrp_kmeans.py` implements:

- Neuralization of KMeans into a neural model
- Layer-wise Relevance Propagation (LRP)
- Feature-level relevance attribution
- Token-level relevance (optional)

The method returns:

- relevance per feature (embedding dimensions)
- relevance per token (if embeddings are provided)

## ▶️ How to Run

Import the model and run it on your data:

```python
from lrp_kmeans import NeuralizedKMeans
from sklearn.cluster import KMeans

# X: your data (N x D)
kmeans = KMeans(n_clusters=4).fit(X)
centers = kmeans.cluster_centers_

model = NeuralizedKMeans(centers)

# Feature-level relevance
R_features = model.explain(X)

# Optional: token-level relevance
R_features, R_tokens = model.explain(X, doc_embeddings)

