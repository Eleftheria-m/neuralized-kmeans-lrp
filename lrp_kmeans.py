import numpy as np
import torch
import torch.nn as nn


class NeuralizedKMeans(nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.centers = centers
        self.num_clusters, self.dim = centers.shape
        self.weights, self.bias = self._compute_params()

        self.linear = nn.Linear(self.dim, self.num_clusters, bias=True)
        self._init_layer()

    def _compute_params(self):
        weights = []
        bias = []

        for i in range(len(self.centers)):
            dists = [np.linalg.norm(self.centers[i] - self.centers[j])
                     for j in range(len(self.centers)) if j != i]
            nearest_j = np.argmin(dists)
            j = [k for k in range(len(self.centers)) if k != i][nearest_j]

            w = 2 * (self.centers[i] - self.centers[j])
            b = np.linalg.norm(self.centers[j])**2 - np.linalg.norm(self.centers[i])**2

            weights.append(w)
            bias.append(b)

        return np.array(weights), np.array(bias)

    def _init_layer(self):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor(self.weights, dtype=torch.float64))
            self.linear.bias = nn.Parameter(torch.tensor(self.bias, dtype=torch.float64))

    def forward(self, x):
        return self.linear(x)

    # ---------- LRP ----------

    @staticmethod
    def second_min(x):
        sorted_arr = np.sort(x, axis=1)
        return sorted_arr[:, 1]

    @staticmethod
    def compute_beta(fc):
        return 1 / np.mean(fc)

    @staticmethod
    def lrp_hidden(outputs, fc, beta):
        e = np.exp(-beta * outputs)
        rk = np.zeros_like(outputs)

        for i in range(outputs.shape[0]):
            c = np.argmin(outputs[i])
            e[i, c] = 0
            d = np.sum(e[i])

            if d != 0:
                rk[i] = (e[i] / d) * fc[i]

        return rk

    @staticmethod
    def compute_midpoints(centers):
        K = centers.shape[0]
        mids = np.zeros((K, K, centers.shape[1]))

        for i in range(K):
            for j in range(K):
                mids[i, j] = (centers[i] + centers[j]) / 2

        return mids

    @staticmethod
    def lrp_input(X, midpoints, weights, rk, clusters):
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()

        N, D = X.shape
        K = midpoints.shape[0]
        R = np.zeros((N, D))
        eps = 1e-9

        for i in range(N):
            c = clusters[i]

            for k in range(K):
                if c == k:
                    continue

                midpoint = midpoints[c, k]
                z = (X[i] - midpoint) * weights[k]
                denom = np.sum(z) + eps

                R[i] += rk[i, k] * (z / denom)

        return R

    @staticmethod
    def distribute_relevance_to_tokens(doc_embeddings, R_input):
        N = len(doc_embeddings)
        D = R_input.shape[1]
        EPS = 1e-9

        docs_relevance = []

        for i in range(N):
            E = np.array(doc_embeddings[i])

            if E.ndim != 2 or E.shape[0] == 0:
                docs_relevance.append(np.array([]))
                continue

            n_i = E.shape[0]
            R_vec = R_input[i]
            R_tokens = np.zeros(n_i)
            sum_e = np.sum(E, axis=0)

            for d in range(D):
                denom = sum_e[d] if abs(sum_e[d]) > EPS else EPS
                for j in range(n_i):
                    R_tokens[j] += R_vec[d] * (E[j, d] / denom)

            docs_relevance.append(R_tokens)

        return docs_relevance

    # ---------- Full pipeline ----------

    def explain(self, X, doc_embeddings=None):
        X_tensor = torch.tensor(X, dtype=torch.float64)

        outputs = self.forward(X_tensor).detach().numpy()
        clusters = np.argmin(outputs, axis=1)

        fc = self.second_min(outputs)
        beta = self.compute_beta(fc)

        rk = self.lrp_hidden(outputs, fc, beta)
        midpoints = self.compute_midpoints(self.centers)

        R_features = self.lrp_input(X, midpoints, self.weights, rk, clusters)

        if doc_embeddings is not None:
            R_tokens = self.distribute_relevance_to_tokens(doc_embeddings, R_features)
            return R_features, R_tokens

        return R_features