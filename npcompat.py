#!/usr/bin/env python3
"""纯 numpy 的替身,让整套实验在只有 numpy 的机器上也能跑,不用装 sklearn/scipy。

   顺带提供 scatter_add —— 用 bincount 重写 np.add.at,这是整个训练里最慢的一步。
   np.add.at 走的是无缓冲的逐元素路径,对 (B*50, K) 这种规模比 bincount 慢一个量级。
"""
import numpy as np


# ---------- 指标(替代 sklearn.metrics)----------
def roc_auc_score(y, p):
    """按秩计算,并列取平均秩 —— 和 sklearn 的定义一致。"""
    y = np.asarray(y).ravel().astype(np.float64)
    p = np.asarray(p).ravel().astype(np.float64)
    o = np.argsort(p, kind='mergesort')
    ps, ys = p[o], y[o]
    n = len(ps)
    rank = np.empty(n, np.float64)
    i = 0
    while i < n:                       # 并列区间取平均秩
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = ys.sum()
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((rank[ys == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def log_loss(y, p, eps=1e-6):
    y = np.asarray(y, np.float64).ravel()
    p = np.clip(np.asarray(p, np.float64).ravel(), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score_loss(y, p):
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    return float(np.mean((p - y) ** 2))


# ---------- 一维截距拟合(替代 scipy.optimize)----------
def fit_intercept(z, y, lo=-3.0, hi=3.0, iters=60):
    """最小化 logloss 关于标量 b。导数 mean(sigmoid(z+b) - y) 关于 b 单调递增,
       所以二分法既简单又绝对稳,不需要 scipy。"""
    z = np.asarray(z, np.float64); y = np.asarray(y, np.float64)
    for _ in range(iters):
        m = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + m))) - y) > 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


# ---------- 热点:scatter add ----------
def scatter_add(dst, idx, val):
    """dst[idx] += val,支持 dst 是 (D,) 或 (D,K)。比 np.add.at 快约一个量级。

       (D,K) 的情形逐列 bincount:bincount 是编译好的紧循环,
       而 np.add.at 是 ufunc.at 的无缓冲慢路径。
    """
    idx = np.asarray(idx).ravel()
    v = np.asarray(val, np.float64)
    if dst.ndim == 1:
        wv = np.broadcast_to(v, idx.shape) if v.size == 1 else v.ravel()
        dst += np.bincount(idx, weights=wv, minlength=dst.shape[0])[:dst.shape[0]].astype(dst.dtype)
        return dst
    V = (np.broadcast_to(v, (idx.size, dst.shape[1])) if v.size == 1
         else np.asarray(val).reshape(-1, dst.shape[1]))
    D = dst.shape[0]
    for k in range(dst.shape[1]):
        dst[:, k] += np.bincount(idx, weights=V[:, k].astype(np.float64),
                                 minlength=D)[:D].astype(dst.dtype)
    return dst


if __name__ == '__main__':
    # 自检:和 numpy/sklearn 的参考实现对齐
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < .5).astype(float)
    p = np.clip(rng.random(5000), .01, .99)
    print('logloss', log_loss(y, p), 'brier', brier_score_loss(y, p), 'auc', roc_auc_score(y, p))
    z = rng.normal(0, 1, 5000); yy = (rng.random(5000) < 1 / (1 + np.exp(-(z + .3)))).astype(float)
    print('截距拟合(真值 +0.30):', round(fit_intercept(z, yy), 3))
    D, K, N = 50, 8, 20000
    idx = rng.integers(0, D, N); val = rng.normal(0, 1, (N, K))
    a = np.zeros((D, K)); np.add.at(a, idx, val)
    b = scatter_add(np.zeros((D, K)), idx, val)
    print('scatter_add 与 np.add.at 最大差:', np.abs(a - b).max())


# ---------- 极简 KMeans(替代 sklearn.cluster.KMeans)----------
class KMeans:
    """k-means++ 初始化 + Lloyd 迭代。只实现实验里用到的接口。"""
    def __init__(self, n_clusters=8, n_init=4, max_iter=100, random_state=0):
        self.k = n_clusters; self.n_init = n_init; self.max_iter = max_iter
        self.rs = random_state; self.labels_ = None; self.cluster_centers_ = None

    def fit(self, X):
        X = np.asarray(X, np.float64); n = len(X)
        rng = np.random.default_rng(self.rs)
        best = (np.inf, None, None)
        for _ in range(self.n_init):
            C = np.empty((self.k, X.shape[1]))
            C[0] = X[rng.integers(n)]
            d2 = ((X - C[0]) ** 2).sum(1)
            for j in range(1, self.k):                 # k-means++
                s = d2.sum()
                C[j] = X[rng.integers(n)] if s <= 0 else X[rng.choice(n, p=d2 / s)]
                d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
            lab = None
            for _ in range(self.max_iter):
                D = ((X[:, None, :] - C[None]) ** 2).sum(2) if n * self.k < 4_000_000 else \
                    (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
                nl = D.argmin(1)
                if lab is not None and (nl == lab).all(): lab = nl; break
                lab = nl
                for j in range(self.k):
                    m = lab == j
                    if m.any(): C[j] = X[m].mean(0)
            inertia = float(((X - C[lab]) ** 2).sum())
            if inertia < best[0]: best = (inertia, lab.copy(), C.copy())
        self.inertia_, self.labels_, self.cluster_centers_ = best
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
