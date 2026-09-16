#!/usr/bin/env python3
"""所有实验共用的评测台。任何新方法都必须走这里,否则结果不可比。

   ⚠️ 唯一允许的输入:seats (n,10,5) —— 10 个座位,每座位 5 件东西的 ID
      (1 个英雄本体 + 4 个技能,顺序不固定,英雄本体在 ids[] 里的值为负)。
      前 5 个座位是一方,后 5 个是另一方。除此之外什么都不许用:
      不许用 windrun 的胜率/选取率/顺位/组合表,不许用 pick order,
      不许用 rating,更不许用任何赛后信息(击杀/经济/时长/物品)。

   固定切分(数据已按开局时间排好序):
      训练 [0, 0.80)   验证 [0.80, 0.85)   测试 [0.85, 1.0)
   验证集只能用来早停和选超参;测试集只在最后 evaluate() 里碰一次。

   模型契约:你提供 score(S) -> z,S 是 (B,10,5) 的 ID 数组,z 是 (B,) 的实数。
   z 必须是**反对称**的:把前 5 座位和后 5 座位整体对调,z 必须变号。
   先手优势(数据里约 53.8%)由 evaluate() 单独拟合一个截距,不要你操心 ——
   模拟器里没有天辉/夜魇,这个截距最终会被扔掉。
"""
import numpy as np
from npcompat import roc_auc_score, log_loss, brier_score_loss, fit_intercept

import os
PURE = os.environ.get('BENCH_PURE', 'data_pure.npz')


def load():
    """返回 (seats, y, ids, (tr, va, te))。"""
    d = np.load(PURE)
    seats, y, ids = d['seats'].astype(np.int32), d['y'].astype(np.float32), d['ids']
    n = len(y)
    if 'ntr' in d.files:                      # 扩展数据集: 切分点随文件走(验证/测试段与原数据集完全相同)
        ntr, nva = int(d['ntr']), int(d['nva'])
        return seats, y, ids, (slice(0, ntr), slice(ntr, nva), slice(nva, n))
    return seats, y, ids, (slice(0, int(n * .80)),
                           slice(int(n * .80), int(n * .85)),
                           slice(int(n * .85), n))


def _chunks(S, fn, bs=4096):
    # 批不能大:K=96 时 (bs,10,5,96) 的中间量就有 bs×0.018 MB,
    # bs=20000 会到 350 MB,几个叠起来直接撞穿 1.4 GB 的 ulimit。
    # 训练用 batch=2048 一直没事,唯独这里的评测批开太大 —— 已经害死过 4 个跑满 2 小时的任务。
    return np.concatenate([np.asarray(fn(S[i:i + bs]), np.float64) for i in range(0, len(S), bs)])


def evaluate(name, score_fn, note='', show_cal=True):
    """唯一的正式评测入口。自动:拟合先手截距 → 出测试集指标 → 校准表 → 反对称自检。"""
    seats, y, ids, (tr, va, te) = load()
    z_tr, z_te = _chunks(seats[tr], score_fn), _chunks(seats[te], score_fn)
    # 先手优势:只用训练集拟合一个标量截距(纯 numpy 二分,不依赖 scipy)
    b = fit_intercept(z_tr, y[tr])
    p = 1 / (1 + np.exp(-(z_te + b)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yte = y[te]
    a, l, br = roc_auc_score(yte, p), log_loss(yte, p), brier_score_loss(yte, p)
    print(f'RESULT\t{name}\tAUC={a:.4f}\tlogloss={l:.4f}\tbrier={br:.4f}\tb={b:+.4f}\t{note}', flush=True)
    unc, res, rel = murphy(yte, p)
    print(f'  Murphy 分解  不确定性 {unc:.4f} − 分辨力 {res:.4f} + 可靠性 {rel:.5f}'
          f'   信号提取率 {res/unc:.2%}', flush=True)
    err = antisym_check(score_fn, seats)
    cal = calibration(yte, p) if show_cal else None
    try:                       # 存逐样本预测,任何两个模型之后都能做配对检验,
        np.save(f'pred_{name}.npy', p.astype(np.float32))   # 不用再手抄一遍前向
    except Exception as ex:
        print('  (预测未能存盘:', ex, ')')
    return {'name': name, 'auc': float(a), 'logloss': float(l), 'brier': float(br),
            'intercept': float(b), 'antisym_err': err, 'max_cal_err': cal,
            'resolution': float(res), 'reliability': float(rel), 'signal_rate': float(res / unc)}


def murphy(y, p, bins=20):
    """Brier = 不确定性 − 分辨力 + 可靠性。
       不确定性是胜负本身的总方差(天花板,动不了);
       分辨力是模型真正抽出来的信号(越大越好,这才是要盯的数);
       可靠性是校准误差(越小越好)。"""
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0] -= 1e-9
    idx = np.clip(np.searchsorted(q, p, 'left') - 1, 0, bins - 1)
    ybar = y.mean(); unc = ybar * (1 - ybar); res = rel = 0.0
    for k in range(bins):
        m = idx == k
        if m.sum() < 20: continue
        nk = m.sum() / len(y); ok = y[m].mean(); pk = p[m].mean()
        res += nk * (ok - ybar) ** 2; rel += nk * (pk - ok) ** 2
    return unc, res, rel


def antisym_check(score_fn, seats, k=512):
    """把左右两队整体对调,z 必须变号。模拟器里没有先手方,这是硬约束。"""
    S = seats[:k]
    Sw = np.concatenate([S[:, 5:], S[:, :5]], 1)
    z1, z2 = np.asarray(score_fn(S), np.float64), np.asarray(score_fn(Sw), np.float64)
    err = float(np.abs(z1 + z2).max())
    print(f'  反对称自检 max|z(A,B)+z(B,A)| = {err:.2e}  {"OK" if err < 1e-3 else "!! 不满足,结果作废"}')
    return err


def calibration(y_true, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (p >= q[i]) & (p < q[i + 1] if i < bins - 1 else p <= q[bins])
        if m.sum() < 30:
            continue
        rows.append((q[i], q[i + 1], int(m.sum()), float(p[m].mean()), float(y_true[m].mean())))
    print('  校准表  预测区间 / 局数 / 平均预测 / 实际 / 偏差')
    for lo, hi, k, pm, ym in rows:
        print(f'    [{lo:.3f},{hi:.3f})  {k:6d}  {pm:.3f}  {ym:.3f}  {ym-pm:+.3f}')
    return max(abs(ym - pm) for *_, pm, ym in rows) if rows else None


if __name__ == '__main__':
    seats, y, ids, (tr, va, te) = load()
    print(f'{len(y):,} 局   训练 {tr.stop:,} / 验证 {va.stop-va.start:,} / 测试 {len(y)-te.start:,}')
    print(f'物品 {len(ids)} 件(英雄本体 {(ids<0).sum()} + 技能 {(ids>0).sum()})')
    print(f'前 5 座位那一方胜率:训练 {y[tr].mean():.4f}  测试 {y[te].mean():.4f}')
