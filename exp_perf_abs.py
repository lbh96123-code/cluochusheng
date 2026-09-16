#!/usr/bin/env python3
"""纯表现模型: 技能组 → 这个座位这局的绝对表现(8 项, 跨队可比), 不碰胜负。
   标签(按分钟归一, 训练段 z-score): gpm, xpm, 补刀/分, 伤害/分, log1p(治疗/分), 击杀/分, 死亡/分, 助攻/分
   模型: 每件东西一根 K 维向量 e + 每件对每项的直接加成 b;
         座位 Ss=Σe, 座位内配合 syn(切半带符号 FM), 队友 Σ, 对手 Σ
         READOUT=linear: p̂ = Σb + [Ss, 队友, 对手, syn]@A + c
         READOUT=mlp:    p̂ = Σb + tanh([Ss, 队友, 对手]@W1+b1)@W2 + syn·a + c
   用法: BENCH_PURE=data_pure_d.npz PERF_RAW=data_perf_raw_d.npz READOUT=linear EXT_TAG=_d python3 exp_perf_abs.py [K] [EP] [LR]
   GRADCHECK=1 做 float64 有限差分自检后退出。"""
import numpy as np, os, sys, time, json
os.chdir('/root/adres/research'); sys.path.insert(0, '.')
from npcompat import scatter_add
K = int(sys.argv[1]) if len(sys.argv) > 1 else 96
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
READOUT = os.environ.get('READOUT', 'linear'); H = int(os.environ.get('H', 64)); PAT = int(os.environ.get('PATIENCE', 4))
SEED = int(os.environ.get('SEED', 0)); L2 = float(os.environ.get('L2', 1e-6)); GC = os.environ.get('GRADCHECK') == '1'
TAG = f'perfabs_K{K}_{READOUT}' + (f'_H{H}' if READOUT == 'mlp' else '') + os.environ.get('EXT_TAG', '')
DT = np.float64 if GC else np.float32
d = np.load(os.environ.get('BENCH_PURE', 'data_pure_d.npz')); S_all = d['seats'].astype(np.int32); n = len(S_all); D = 640
if 'ntr' in d.files: TR, VA = int(d['ntr']), int(d['nva'])
else: TR, VA = int(n * .8), int(n * .85)
raw = np.load(os.environ.get('PERF_RAW', 'data_perf_raw_d.npz')); fl = [str(x) for x in raw['fields']]; R = raw['raw']; mins = (raw['dur'] / 60.0).astype(np.float32)[:, None]
g = lambda f: R[:, :, fl.index(f)].astype(np.float32)
TGT = ['gpm', 'xpm', '补刀/分', '伤害/分', '治疗/分(log)', '击杀/分', '死亡/分', '助攻/分']
Y = np.stack([g('gpm'), g('xpm'), g('lastHits') / mins, g('heroDamage') / mins, np.log1p(g('heroHealing') / mins), g('kills') / mins, g('deaths') / mins, g('assists') / mins], 2)
del R, raw
MU = Y[:TR].reshape(-1, 8).mean(0); SD = Y[:TR].reshape(-1, 8).std(0) + 1e-9; Y = ((Y - MU) / SD).astype(np.float16); F = 8
sgn_dim = np.ones(K, DT); sgn_dim[K // 2:] = -1.0
if GC: S_all, Y, n = S_all[:16], Y[:16].astype(np.float64), 16
rng = np.random.default_rng(SEED)
e = rng.normal(0, .02, (D, K)).astype(DT); b = np.zeros((D, F), DT); c = np.zeros(F, DT)
if READOUT == 'linear':
    A = rng.normal(0, .01, (3 * K + 1, F)).astype(DT); PS = [e, b, c, A]
else:
    W1 = rng.normal(0, .05, (3 * K, H)).astype(DT); b1 = np.zeros(H, DT); W2 = rng.normal(0, .05, (H, F)).astype(DT); a_syn = np.zeros(F, DT); PS = [e, b, c, W1, b1, W2, a_syn]
def team_ops(x):
    m = x.shape[0]; xt = x.reshape(m, 2, 5, -1); ts = xt.sum(2, keepdims=True)
    tm = (ts - xt).reshape(m, 10, -1); en = np.repeat(ts[:, ::-1], 5, axis=2).reshape(m, 10, -1); return tm, en
def fwd(Sb):
    E = e[Sb]; Ss = E.sum(2); sq = (E ** 2 * sgn_dim).sum(3).sum(2); syn = .5 * ((Ss ** 2 * sgn_dim).sum(2) - sq)
    tm, en = team_ops(Ss); X = np.concatenate([Ss, tm, en], 2); bs = b[Sb].sum(2)
    if READOUT == 'linear':
        feat = np.concatenate([X, syn[..., None]], 2); p = bs + feat @ A + c; cache = (E, Ss, syn, feat)
    else:
        pre = X @ W1 + b1; h = np.tanh(pre); p = bs + h @ W2 + syn[..., None] * a_syn + c; cache = (E, Ss, syn, X, h)
    return p, cache
def loss_grad(Sb, yb):
    p, cache = fwd(Sb); m = len(Sb); r = p - yb; L = float((r ** 2).mean()); dp = (2 * r / r.size).astype(DT)
    gc_ = dp.sum((0, 1)); gb = np.zeros_like(b); scatter_add(gb, np.repeat(Sb.reshape(-1), 1), np.repeat(dp[:, :, None, :], 5, axis=2).reshape(-1, F))
    if READOUT == 'linear':
        E, Ss, syn, feat = cache
        gA = feat.reshape(-1, 3 * K + 1).T @ dp.reshape(-1, F); gf = dp @ A.T; gX = gf[..., :3 * K]; dsyn = gf[..., 3 * K]; grads_head = [gA]
    else:
        E, Ss, syn, X, h = cache
        gW2 = h.reshape(-1, H).T @ dp.reshape(-1, F); ga = (dp * syn[..., None]).sum((0, 1)); gh = dp @ W2.T; gpre = gh * (1 - h ** 2)
        gW1 = X.reshape(-1, 3 * K).T @ gpre.reshape(-1, H); gb1 = gpre.sum((0, 1)); gX = gpre @ W1.T; dsyn = dp @ a_syn; grads_head = [gW1, gb1, gW2, ga]
    gSs = gX[..., :K].copy(); g_tm = gX[..., K:2 * K]; g_en = gX[..., 2 * K:]
    tm_adj, en_adj = team_ops(g_tm)[0], team_ops(g_en)[1]   # 队友项伴随 = 队友的 g_tm 之和; 对手项伴随 = 对方全队 g_en 之和
    gSs += tm_adj + en_adj + dsyn[..., None] * Ss * sgn_dim
    gE = gSs[:, :, None, :] - dsyn[..., None, None] * E * sgn_dim
    ge = np.zeros_like(e); scatter_add(ge, Sb.reshape(-1), gE.reshape(-1, K)); ge += 2 * L2 * e * m
    return L, [ge, gb, gc_] + grads_head
if GC:
    Sb, yb = S_all[:6], Y[:6]; L0, G = loss_grad(Sb, yb); worst = 0
    for pi, (P_, Gp) in enumerate(zip(PS, G)):
        idx = [np.unravel_index(k, P_.shape) for k in np.random.default_rng(pi).choice(P_.size, min(5, P_.size), replace=False)]
        if pi == 0: idx = [(int(Sb[0, 0, 0]), 1), (int(Sb[2, 7, 3]), K - 1), (int(Sb[4, 3, 2]), K // 2)]
        if pi == 1: idx = [(int(Sb[0, 0, 0]), 0), (int(Sb[2, 7, 3]), 5)]
        for ix in idx:
            old = P_[ix]; h_ = 1e-6; P_[ix] = old + h_; Lp = loss_grad(Sb, yb)[0]; P_[ix] = old - h_; Lm = loss_grad(Sb, yb)[0]; P_[ix] = old
            num = (Lp - Lm) / (2 * h_); ana = Gp[ix] - (2 * L2 * e[ix] * len(Sb) if pi == 0 else 0)
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12); worst = max(worst, rel)
    print(f'梯度自检 READOUT={READOUT}: 最大相对误差 {worst:.2e}  {"OK" if worst < 1e-3 else "!! 不对"}'); sys.exit(0)
mo = [np.zeros_like(p) for p in PS]; ve = [np.zeros_like(p) for p in PS]; t = 0; BS = 2048
def predict(lo, hi):
    P = np.empty((hi - lo, 10, F), np.float32)
    for s in range(lo, hi, 4096): P[s - lo:min(s + 4096, hi) - lo] = fwd(S_all[s:min(s + 4096, hi)])[0]
    return P
best = (1e9, None, 0); t0 = time.time()
for ep in range(EP):
    t1 = time.time(); perm = rng.permutation(TR)
    for s in range(0, TR, BS):
        j = np.sort(perm[s:s + BS]); L, G = loss_grad(S_all[j], Y[j].astype(np.float32)); t += 1
        for p_, g_, m_, v_ in zip(PS, G, mo, ve):
            m_ *= .9; m_ += .1 * g_; v_ *= .999; v_ += .001 * g_ * g_
            p_ -= LR * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8)
    vl = float(((predict(TR, VA) - Y[TR:VA].astype(np.float32)) ** 2).mean())
    print('ep %2d 验证 MSE %.4f  %.0fs' % (ep + 1, vl, time.time() - t1), flush=True)
    if vl < best[0] - 1e-5: best = (vl, [p.copy() for p in PS], ep + 1)
    elif ep + 1 - best[2] >= PAT: break
for p_, q_ in zip(PS, best[1]): p_[...] = q_
print('早停第 %d 轮' % best[2])
# ---------- 评测(测试段) ----------
Pt = predict(VA, n); Yt = Y[VA:].astype(np.float32)
cors = [float(np.corrcoef(Pt[:, :, j].ravel(), Yt[:, :, j].ravel())[0, 1]) for j in range(F)]
r2 = [float(1 - ((Pt[:, :, j] - Yt[:, :, j]) ** 2).mean() / Yt[:, :, j].var()) for j in range(F)]
print('测试段 相关: ' + '  '.join('%s %.3f' % (a, v) for a, v in zip(TGT, cors)))
print('测试段 R²:   ' + '  '.join('%s %.3f' % (a, v) for a, v in zip(TGT, r2)))
# 跨队同位置比较: 每队预测 gpm 最高的座位当"刷钱位", 预测谁的刷钱位 gpm 高 vs 实际
res = {}
for nm, j in (('刷钱位(gpm)', 0), ('输出位(伤害)', 3)):
    PL, PR = Pt[:, :5, j], Pt[:, 5:, j]; iL, iR = PL.argmax(1), PR.argmax(1); ar = np.arange(len(Pt))
    pd_ = PL[ar, iL] - PR[ar, iR]; ad = Yt[ar, iL, j] - Yt[ar, 5 + iR, j]
    acc = float((np.sign(pd_) == np.sign(ad)).mean()); cc = float(np.corrcoef(pd_, ad)[0, 1])
    q = np.abs(pd_) > np.quantile(np.abs(pd_), .8); acc_conf = float((np.sign(pd_[q]) == np.sign(ad[q])).mean())
    res[nm] = dict(acc=acc, corr=cc, acc_top20=acc_conf)
    print('%s 跨队比较: 预测差与实际差 相关 %.3f, 方向判对 %.1f%%, 最有把握的 20%% 判对 %.1f%%' % (nm, cc, 100 * acc, 100 * acc_conf))
# 顺带: 只用表现预测能拿到多少胜负(左−右 8 项队总和 → 逻辑回归), 用户不关心, 仅记录
y_all = d['y'].astype(np.float64)
Ptr = predict(0, TR)
def tf(P): return (P[:, :5].sum(1) - P[:, 5:].sum(1)).astype(np.float64)
Xtr, Xte = tf(Ptr), tf(Pt); del Ptr; w = np.zeros(F); b0 = 0.0
for _ in range(30):
    z = Xtr @ w + b0; p = 1 / (1 + np.exp(-z)); gr = Xtr.T @ (p - y_all[:TR]) + 1.0 * w; Hm = (Xtr * (p * (1 - p))[:, None]).T @ Xtr + 1.0 * np.eye(F)
    w -= np.linalg.solve(Hm, gr); b0 -= (p - y_all[:TR]).sum() / (p * (1 - p)).sum()
pte = np.clip(1 / (1 + np.exp(-(Xte @ w + b0))), 1e-9, 1 - 1e-9); yte = y_all[VA:]
ll_perf = float(-(yte * np.log(pte) + (1 - yte) * np.log(1 - pte)).mean()); acc_w = float(((pte > .5) == (yte > .5)).mean())
print('(参考) 只用表现预测推胜负: 测试 logloss %.5f 方向判对 %.2f%%' % (ll_perf, 100 * acc_w))
# 每件东西对 gpm / 伤害 的加成(直接项 + 经由座位向量的线性读出), 给人看
names = [str(x) for x in np.load('pairaudit/model_pair.npz', allow_pickle=True)['names']]
if READOUT == 'linear': eff = b + e @ A[:K]
else: eff = b.copy()
top = {}
for nm, j in (('gpm', 0), ('伤害/分', 3), ('治疗/分', 4)):
    o = np.argsort(-eff[:, j]); top[nm] = dict(top=[(names[i], float(eff[i, j] * SD[j])) for i in o[:12]], bottom=[(names[i], float(eff[i, j] * SD[j])) for i in o[-8:]])
    print('%s 加成最高: %s' % (nm, ', '.join('%s %+.0f' % x for x in top[nm]['top'][:8])))
os.makedirs('perfabs', exist_ok=True)
np.savez_compressed(f'perfabs/model_{TAG}.npz', **{f'p{i}': p for i, p in enumerate(PS)}, MU=MU, SD=SD, TGT=np.array(TGT), K=K, readout=READOUT, eff=eff)
np.save(f'perfabs/pred_te_{TAG}.npy', Pt.astype(np.float16))
json.dump(dict(tag=TAG, val_mse=best[0], ep=best[2], cors=cors, r2=r2, cross=res, win_ll=ll_perf, win_acc=acc_w, top=top), open(f'perfabs/result_{TAG}.json', 'w'), ensure_ascii=False, indent=1)
print('用时 %.0fs' % (time.time() - t0))
