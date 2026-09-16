#!/usr/bin/env python3
# 输入只读 data_pure.npz(ID);打钱轴由 data_perf 估出后**冻结成常量表**,推理时只是按 ID 查表
"""多任务 + 队伍构成项(+ 可选:带符号配合)。

   SIGNED=1 时配合项改成  Σ_k s_k·e_ik·e_jk。
   动机:普通 FM 的配合矩阵 e·eᵀ **天然半正定**,特征值只能 ≥0 ——
   结构上无法表达"这两个方向互相拖累"。实测把 e 拆成 p/q 之后,
   对称成分有 35.6% 的能量落在负特征值上(前 12 个主方向里 5 个为负),
   说明这个能力是被真实用到的。给每维一个可正可负的 s_k 就够了,
   不必像 p/q 那样把参数翻倍(61,536 vs 122,880)。

   动机(实测):把队伍里"打钱位"的个数从 1 个堆到 4 个,实际胜率比模型预测低 6.7 个百分点
   (t=-13.9);换成治疗轴做同样的事只有 0.75pp。现有 logit 是 10 个座位线性相加,
   结构上表达不了这种边际衰减。

   做法:每件东西一个**冻结**的打钱载荷 fw[i](只用输方数据估,免 collider;只用训练段,免泄漏),
        队伍打钱分 s_T = Σ_{座位∈T} Σ_{件∈座位} fw,
        logit += f(s_L) - f(s_R),f = 分段线性,只有 BINS+1 个自由参数。
   写成差的形式,反对称是结构保证的,不靠训练。

   用法: python3 exp_comp.py [K] [EP] [LR] [LAMBDA] [BINS]
         BINS=0 时构成项关闭,退化成 exp_multitask —— 干净的消融对照。
"""
import numpy as np, sys, time, json, os
import bench
from npcompat import log_loss, scatter_add

K    = int(sys.argv[1]) if len(sys.argv) > 1 else 96
EP   = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LR   = float(sys.argv[3]) if len(sys.argv) > 3 else 8e-4
LAM  = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
BINS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
WITHIN = os.environ.get('WITHIN') == '1'
SEED   = int(os.environ.get('SEED', '0'))     # 集成用:换种子重训,只改随机初值/打乱顺序
SIGNED = os.environ.get('SIGNED') == '1'      # 带符号配合:让配合矩阵可以不定
PERFMODE = os.environ.get('PERFMODE', 'z')          # z = 原 perflabels 八项(全局 z-score) / share = 队内份额口径(M1 同款)
WINCTRL = os.environ.get('WINCTRL') == '1'          # 辅助头加 "本座位赢没赢 × 座位向量" 控制列: 让"赢了核心多拿钱"这种效应不进嵌入
CREDIT = float(os.environ.get('CREDIT_BETA', '0'))
TAG = f'comp_K{K}_lam{LAM}_b{BINS}' + ('_sgn' if SIGNED else '') + ('_win' if WITHIN else '') + (f'_s{SEED}' if SEED else '') + os.environ.get('EXT_TAG', '') + ('_ps' if os.environ.get('PATCHSPLIT') == '1' else '') + (f'_pf{PERFMODE}' if PERFMODE != 'z' else '') + ('_wc' if WINCTRL else '') + (f'_cr{CREDIT}' + ('s' if os.environ.get("CREDIT_SIGN", "1") == "0" else '') if CREDIT > 0 else '')
rng = np.random.default_rng(SEED)

seats, y, ids, (tr, va, te) = bench.load()
D, n = len(ids), len(y)
sgn = np.array([1.] * 5 + [-1.] * 5, np.float32)

import perflabels
PERF, PNAMES = perflabels.build(os.environ.get('PERF_RAW', 'data_perf_raw.npz'))
if PERFMODE == 'share':
    # 队内份额口径(与 m1_perf.py 一字不差): gpm/xpm/补刀/伤害/治疗/死亡 = 占本队份额−0.2; 参战率/KDA = 队内中心化。
    # 份额天然扣掉了"这队赢了/这局长"的队级效应, 剩下的才是"这个座位在队里承担了多少"。
    _raw = np.load(os.environ.get('PERF_RAW', 'data_perf_raw.npz')); _fl = [str(x) for x in _raw['fields']]; _R = _raw['raw']
    _team = lambda x: x.reshape(-1, 2, 5)
    def _share(v):
        tt = _team(v); return (tt / np.maximum(tt.sum(2, keepdims=True), 1e-9)).reshape(-1, 10) - 0.2
    def _tcen(v):
        tt = _team(v); return (tt - tt.mean(2, keepdims=True)).reshape(-1, 10)
    _g = lambda f: _R[:, :, _fl.index(f)].astype(np.float32)
    _Kk, _Dd, _Aa = _g('kills'), _g('deaths'), _g('assists'); _teamK = np.repeat(_team(_Kk).sum(2, keepdims=True), 5, axis=2).reshape(-1, 10)
    PNAMES = ['gpm份额', 'xpm份额', '补刀份额', '伤害份额', '治疗份额', '参战率', '死亡份额', 'KDA']
    PERF = np.stack([_share(_g('gpm')), _share(_g('xpm')), _share(_g('lastHits')), _share(_g('heroDamage')), _share(_g('heroHealing')),
                     _tcen((_Kk + _Aa) / np.maximum(_teamK, 1)), _share(_Dd), _tcen(np.log1p((_Kk + _Aa) / np.maximum(_Dd, 1)))], 2).astype(np.float32)
    del _R, _raw
    PERF /= (PERF[tr].reshape(-1, 8).std(0) + 1e-9)       # 只用训练段的尺度
    print('表现标签口径 = 队内份额(share), 训练段标准化', flush=True)
MARGIN = None
if PERFMODE == 'margin':
    # 队伍优势幅度: 左队总量 − 右队总量(累积量按分钟归一), 五项各自按训练段标准化后取均值, 再标准化。反对称, 与 z 同号约定。
    _raw = np.load(os.environ.get('PERF_RAW', 'data_perf_raw.npz')); _fl = [str(x) for x in _raw['fields']]; _R = _raw['raw'].astype(np.float32); _mins = (_raw['dur'] / 60.0).astype(np.float32)
    _g = lambda f: _R[:, :, _fl.index(f)]
    _ds = []
    for f, rate in (('gpm', False), ('xpm', False), ('lastHits', True), ('heroDamage', True), ('kills', True)):
        v = _g(f); d = v[:, :5].sum(1) - v[:, 5:].sum(1)
        if rate: d = d / _mins
        _ds.append(d / (d[tr].std() + 1e-9))
    MARGIN = np.mean(_ds, 0); MARGIN = (MARGIN / (MARGIN[tr].std() + 1e-9)).astype(np.float32); del _R, _raw
    print('队伍优势幅度标签: 与胜负相关 %.3f, 赢方均值 %+.2f' % (np.corrcoef(MARGIN[tr], y[tr])[0, 1], MARGIN[tr][y[tr] > .5].mean()), flush=True)

PW = None
if PERFMODE in ('comp', 'abs8w'):
    # 绝对表现 8 项(按分钟归一, 与 exp_perf_abs.py 同口径), 训练段 z; 不做队内中心化(跨队可比)
    _raw = np.load(os.environ.get('PERF_RAW', 'data_perf_raw.npz')); _fl = [str(x) for x in _raw['fields']]; _R = _raw['raw']; _mins = (_raw['dur'] / 60.0).astype(np.float32)[:, None]
    _g = lambda f: _R[:, :, _fl.index(f)].astype(np.float32)
    _Y = np.stack([_g('gpm'), _g('xpm'), _g('lastHits') / _mins, _g('heroDamage') / _mins, np.log1p(_g('heroHealing') / _mins), _g('kills') / _mins, _g('deaths') / _mins, _g('assists') / _mins], 2); del _R, _raw
    _mu = _Y[tr].reshape(-1, 8).mean(0); _sd = _Y[tr].reshape(-1, 8).std(0) + 1e-9; _Z = ((_Y - _mu) / _sd).astype(np.float32); del _Y
    import json as _json
    if PERFMODE == 'comp':
        _w8 = np.array(_json.load(open('perfabs/quick_kda.json'))['w'], np.float32)      # A 权重: 真实 8 项(座位级)→本队胜负
        PERF = (_Z @ _w8)[..., None]; PERF = ((PERF - PERF[tr].mean()) / (PERF[tr].std() + 1e-9)).astype(np.float32); PNAMES = ['综合分']
        print('表现标签 = 一维 STRATZ 式综合分, 权重', np.round(_w8, 2), flush=True)
    else:
        PERF = _Z; PNAMES = ['gpm', 'xpm', '补刀/分', '伤害/分', '治疗/分', '击杀/分', '死亡/分', '助攻/分']
        PW = np.abs(np.array(_json.load(open('perfabs/adjw.json'))['wC'], np.float32)); PW = (PW / PW.mean()).astype(np.float32)   # C 权重: 预测 8 项两队差→胜负
        print('表现标签 = 绝对 8 项, 逐项损失权重(|C| 归一):', dict(zip(PNAMES, np.round(PW, 2))), flush=True)
    del _Z
PERF = PERF.astype(np.float16)

CW = None
if CREDIT > 0:
    # 按表现分功劳: 座位功劳权重 = clip(1 + β·发挥分·(赢+1/输−1), 0, 2); 发挥分 = 实际综合分 − 选人预期综合分(perfabs/credit_f_d.npy)
    _f = np.load(os.environ.get('CREDIT_F', 'perfabs/credit_f_d.npy')).astype(np.float32); assert len(_f) == n
    _st = (sgn[None, :] * (2.0 * y - 1.0)[:, None]).astype(np.float32)
    if os.environ.get("CREDIT_SIGN", "1") == "0": _st = np.ones_like(_st)   # 对称版: 权重不看输赢, 打得活跃的座位多承担本局结果(赢或输), 无偏
    CW = np.clip(1.0 + CREDIT * _f * _st, 0.0, 2.0).astype(np.float16); del _f, _st
    print('按表现分功劳 β=%.2f: 权重均值 %.3f 标准差 %.3f, 赢方最高/最低 %.2f/%.2f' % (CREDIT, CW.astype(np.float32).mean(), CW.astype(np.float32).std(), CW[:1000].astype(np.float32).max(), CW[:1000].astype(np.float32).min()), flush=True)
assert len(PERF) == n
F = PERF.shape[2]
# 赢没赢(座位视角 ±1): 只在训练辅助头时用, 推理不需要
WS = (sgn[None, :] * (2.0 * y - 1.0)[:, None]).astype(np.float32) if WINCTRL else None
NF = (4 * K + 2) if WINCTRL else (3 * K + 2)

# ---------- 打钱轴:冻结常量,只用训练段 + 只用输方 ----------
def farm_axis():
    raw = np.load(os.environ.get('PERF_Z', 'data_perf.npz'))
    v = raw['z'][:, :, list(raw['fields']).index('lastHits')].astype(np.float64)   # (n,10)
    lost = np.zeros((n, 10), bool)
    lost[:, :5] = (y == 0)[:, None]; lost[:, 5:] = (y == 1)[:, None]   # 输的那一方
    use = lost.copy(); use[tr.stop:] = False                            # 只用训练段
    tot = np.zeros(D); cnt = np.zeros(D)
    for k in range(5):
        s = seats[:, :, k]; ok = (s >= 0) & use
        scatter_add(tot, s[ok], v[ok]); scatter_add(cnt, s[ok], 1.0)
    fw = tot / np.maximum(cnt, 1)
    fw[cnt < 50] = 0.0                       # 样本太少的物品不给载荷,免噪声
    return fw.astype(np.float32)

fw = farm_axis()
def team_scores(S):
    f = fw[S]                                             # (B,10,5)
    return f[:, :5].sum((1, 2)), f[:, 5:].sum((1, 2))     # sL, sR

if BINS > 0:
    _sl, _sr = team_scores(seats[tr])
    KNOT = np.quantile(np.concatenate([_sl, _sr]), np.linspace(0, 1, BINS + 1)).astype(np.float32)
    KNOT[0] -= 1e-3; KNOT[-1] += 1e-3
    print(f'打钱轴:非零载荷 {int((fw!=0).sum())}/{D} 件   分段结点 '
          + ' '.join(f'{x:.2f}' for x in KNOT), flush=True)

def bin_wt(s):
    """s -> (下标 j, 权重 frac):f(s) = θ[j]*(1-frac) + θ[j+1]*frac"""
    j = np.clip(np.searchsorted(KNOT, s) - 1, 0, BINS - 1)
    frac = (s - KNOT[j]) / (KNOT[j + 1] - KNOT[j])
    return j, np.clip(frac, 0, 1).astype(np.float32)

w = np.zeros(D, np.float32)
# ---- 单件按版本分开(配合 e / 配比 th / 辅助头 共享): z 里的单件 = w[S] + dw[patch, S]; 7.41d 的 dw 恒为 0 ----
PATCHSPLIT = os.environ.get('PATCHSPLIT') == '1'
PI = np.zeros(n, np.int32); NPATCH = 1
if PATCHSPLIT:
    _pz = np.load(os.environ.get('BENCH_PURE', 'data_pure.npz'))
    _pl = [str(x) for x in _pz['patch']]; _names = sorted(set(_pl) - {'orig', '7.41d'})
    _map = {'orig': 0, '7.41d': 0, **{nm: i + 1 for i, nm in enumerate(_names)}}
    PI = np.array([_map[x] for x in _pl], np.int32); NPATCH = 1 + len(_names)
    print('单件分版本:', {nm: int((PI == i).sum()) for nm, i in _map.items()}, flush=True)
dw = np.zeros((NPATCH, D), np.float32)
e = rng.normal(0, .005, (D, K)).astype(np.float32)
A = rng.normal(0, .01, (NF, F)).astype(np.float32)
c = np.zeros(F, np.float32)
th = np.zeros(BINS + 1, np.float32) if BINS > 0 else np.zeros(1, np.float32)
# 符号**固定**不学:前 K/2 维加分、后 K/2 维减分。
# 这样配合矩阵 E·diag(s)·Eᵀ 允许负特征值 —— 才说得出"同类扎堆反而亏"。
# 实测 K=192 对称(参数同样翻倍)毫无提升,所以缺的不是容量,是这个符号。
SGN = np.ones(K, np.float32)
if SIGNED: SGN[K // 2:] = -1.0
ab = np.array([1.0, 0.0], np.float32)          # margin 模式: 预测幅度 = ab[0]·z + ab[1]
PS = [w, e, A, c, th, dw, ab]
mo = [np.zeros_like(p) for p in PS]; ve = [np.zeros_like(p) for p in PS]
b1, b2, eps = .9, .999, 1e-8
LR_TH = LR * 3.0                      # θ 只有 9 个参数、从 0 起步,给大一点的步长
t = 0; best = (1e9, None, 0)
PAT = int(os.environ.get('PATIENCE', '6'))    # 连续这么多轮没进步就停 —— 之前没加,K=192 白跑了 20 多轮

def ts(x):
    B, _, KK = x.shape
    return np.repeat(x.reshape(B, 2, 5, KK).sum(2), 5, axis=1)
def tsw(x):
    B, _, KK = x.shape
    return np.repeat(x.reshape(B, 2, 5, KK).sum(2)[:, ::-1], 5, axis=1)

def comp_term(S):
    if BINS == 0: return 0.0, None
    sL, sR = team_scores(S)
    jL, fL = bin_wt(sL); jR, fR = bin_wt(sR)
    fz = (th[jL] * (1 - fL) + th[jL + 1] * fL) - (th[jR] * (1 - fR) + th[jR + 1] * fR)
    return fz, (jL, fL, jR, fR)

def fwd(S, pi=None):
    E = e[S]; Ss = E.sum(2)
    sq = (E ** 2 * SGN).sum(3)                       # 带符号的平方和
    syn = .5 * ((Ss ** 2 * SGN).sum(2) - sq.sum(2))  # SGN 全 1 时退化回原式
    ws = w[S] if pi is None else w[S] + dw[pi[:, None, None], S]
    z = ((ws.sum(2) + syn) * sgn).sum(1)
    fz, aux = comp_term(S)
    return z + fz, E, Ss, sq, syn, aux

def perf_feat(Ss, syn, sq, ws=None):
    cols = [Ss, ts(Ss), tsw(Ss), syn[..., None], sq.sum(2)[..., None]]
    if ws is not None: cols.append(ws[..., None] * Ss)       # 赢没赢 × 座位向量: 赢方核心多拿的那份归这一列, 不归嵌入
    return np.concatenate(cols, axis=2)

def score(S):
    return fwd(np.asarray(S, np.int32))[0]

# ---------- float64 有限差分自检 ----------
if os.environ.get('GRADCHECK') == '1' and BINS > 0:
    import copy
    S = seats[:64].astype(np.int32); yy = y[:64].astype(np.float64)
    def loss_th(thv):
        old = th.copy(); th[:] = thv
        z = fwd(S)[0].astype(np.float64)
        th[:] = old
        p = 1/(1+np.exp(-z)); return float(-(yy*np.log(p)+(1-yy)*np.log(1-p)).mean())
    th64 = th.astype(np.float64)
    z, E, Ss, sq, syn, aux = fwd(S)
    g = (1/(1+np.exp(-z)) - yy)/len(S)
    jL, fL, jR, fR = aux
    ana = np.zeros(BINS+1)
    scatter_add(ana, jL,   g*(1-fL));  scatter_add(ana, jL+1, g*fL)
    scatter_add(ana, jR,  -g*(1-fR));  scatter_add(ana, jR+1, -g*fR)
    h = 1e-5; num = np.zeros(BINS+1)
    for i in range(BINS+1):
        a = th64.copy(); a[i] += h; b = th64.copy(); b[i] -= h
        num[i] = (loss_th(a.astype(np.float32)) - loss_th(b.astype(np.float32)))/(2*h)
    rel = np.abs(ana-num)/np.maximum(np.abs(num), 1e-9)
    print(f'梯度自检 θ: 最大相对误差 {rel.max():.2e}  {"OK" if rel.max()<2e-2 else "!! 不对"}', flush=True)
    sys.exit(0)

if os.environ.get('GRADCHECK2') == '1':
    """隔离 float64 自检:只看配合项那一路对 e 的梯度(SIGNED 的符号很容易接错)。
       整个 z 的梯度被 /B 压到 1e-6,float32 差分测到的全是浮点噪声 —— 必须隔离 + float64。"""
    S = seats[:64].astype(np.int64); yy = y[:64].astype(np.float64)
    e64 = e.astype(np.float64); SG = SGN.astype(np.float64)
    def sfwd(ev):
        Ee = ev[S]; Ss_ = Ee.sum(2)
        sq_ = (Ee ** 2 * SG).sum(3)
        syn_ = .5 * ((Ss_ ** 2 * SG).sum(2) - sq_.sum(2))
        return (syn_ * sgn).sum(1), Ee, Ss_
    def sloss(ev):
        z_, _, _ = sfwd(ev); pr = 1/(1+np.exp(-z_))
        return float(-(yy*np.log(pr)+(1-yy)*np.log(1-pr)).mean())
    z_, Ee, Ss_ = sfwd(e64)
    g_ = ((1/(1+np.exp(-z_)) - yy)/len(S))[:, None] * sgn[None, :]
    gE_ = (g_[..., None] * Ss_ * SG)[:, :, None, :] - g_[..., None, None] * Ee * SG
    ana = np.zeros_like(e64); scatter_add(ana, S.ravel(), gE_.reshape(-1, K))
    hit = np.argsort(-np.abs(ana).sum(1))[:5]; errs = []
    for i_ in hit:
        for k_ in (0, 1, K//2, K-1):
            a1 = e64.copy(); a1[i_, k_] += 1e-6
            a2 = e64.copy(); a2[i_, k_] -= 1e-6
            num = (sloss(a1) - sloss(a2))/2e-6
            errs.append(abs(ana[i_, k_]-num)/max(abs(num), 1e-12))
    em = max(errs)
    print(f'梯度自检 e(隔离/float64,SIGNED={int(SIGNED)}): 最大相对误差 {em:.2e}  '
          f'{"OK" if em < 1e-2 else "!! 不对"}', flush=True)
    sys.exit(0)

if os.environ.get('GRADCHECK3') == '1':
    """隔离 float64 自检: 只看辅助头那一路(含 WINCTRL 列与 WITHIN 中心化)对 e / A 的梯度。"""
    ix = np.arange(48); S = seats[ix].astype(np.int64); Pb = PERF[ix].astype(np.float64); B = len(ix)
    e64 = e.astype(np.float64); A64 = A.astype(np.float64) * 30; c64 = c.astype(np.float64); SG = SGN.astype(np.float64)
    wsb = WS[ix].astype(np.float64) if WINCTRL else None
    def aux_loss(ev, Av):
        E = ev[S]; Ss_ = E.sum(2); sq_ = (E ** 2 * SG).sum(3); syn_ = .5 * ((Ss_ ** 2 * SG).sum(2) - sq_.sum(2))
        feat = perf_feat(Ss_, syn_, sq_, wsb); r = feat @ Av + c64 - Pb
        if WITHIN: r = r - r.mean(1, keepdims=True)
        return LAM * ((r ** 2) * (PW if PW is not None else 1.0)).mean(), (E, Ss_, sq_, syn_, feat, r)
    L0, (E, Ss_, sq_, syn_, feat, r) = aux_loss(e64, A64)
    gp = 2.0 * LAM * r * (PW if PW is not None else 1.0) / (B * 10 * F)
    gA_ = feat.reshape(-1, NF).T @ gp.reshape(-1, F); gf = gp @ A64.T
    gSs_d = gf[..., :K] + ts(gf[..., K:2 * K]) + tsw(gf[..., 2 * K:3 * K])
    if WINCTRL: gSs_d = gSs_d + wsb[..., None] * gf[..., 3 * K + 2:4 * K + 2]
    gsyn_ = gf[..., 3 * K]; gsq_ = -.5 * gsyn_[..., None] * np.ones((1, 1, 5)) + gf[..., 3 * K + 1][..., None]
    gSs_ = gSs_d + gsyn_[..., None] * Ss_ * SG
    gE = gSs_[:, :, None, :] + 2.0 * gsq_[..., None] * E * SG
    ge_ = np.zeros_like(e64); scatter_add(ge_, S.ravel(), gE.reshape(-1, K))
    errs = []
    for i_ in np.argsort(-np.abs(ge_).sum(1))[:4]:
        for k_ in (0, 1, K // 2, K - 1):
            a1 = e64.copy(); a1[i_, k_] += 1e-6; a2 = e64.copy(); a2[i_, k_] -= 1e-6
            num = (aux_loss(a1, A64)[0] - aux_loss(a2, A64)[0]) / 2e-6
            errs.append(abs(ge_[i_, k_] - num) / max(abs(num), 1e-12))
    errsA = []
    for (i_, j_) in [(0, 0), (K, 3 % F), (3 * K, 5 % F), (3 * K + 1, 7 % F), (NF - 1, 2 % F), (NF - K // 2, 4 % F)]:
        a1 = A64.copy(); a1[i_, j_] += 1e-6; a2 = A64.copy(); a2[i_, j_] -= 1e-6
        num = (aux_loss(e64, a1)[0] - aux_loss(e64, a2)[0]) / 2e-6
        errsA.append(abs(gA_[i_, j_] - num) / max(abs(num), 1e-12))
    print(f'梯度自检 辅助头(float64, PERFMODE={PERFMODE} WINCTRL={int(WINCTRL)} WITHIN={int(WITHIN)}): e 最大相对误差 {max(errs):.2e}  A 最大相对误差 {max(errsA):.2e}  '
          f'{"OK" if max(errs) < 1e-2 and max(errsA) < 1e-2 else "!! 不对"}', flush=True)
    sys.exit(0)

ntr = tr.stop
for ep in range(EP):
    t0 = time.time(); order = rng.permutation(ntr)
    for st in range(0, ntr, 2048):
        ix = order[st:st + 2048]; S = seats[ix]; B = len(ix)
        pi = PI[ix]; z, E, Ss, sq, syn, aux = fwd(S, pi)
        g = ((1 / (1 + np.exp(-z)) - y[ix]) / B).astype(np.float32)
        gab = np.zeros_like(ab)
        if MARGIN is not None and LAM > 0:
            rm = ab[0] * z + ab[1] - MARGIN[ix]                    # 幅度残差
            g = g + (2.0 * LAM * ab[0] * rm / B).astype(np.float32)   # 经由 z 回传, 与胜负梯度同路
            gab[0] = 2.0 * LAM * (rm * z).sum() / B; gab[1] = 2.0 * LAM * rm.sum() / B
        gs = g[:, None] * sgn[None, :]
        if CW is not None: gs = gs * CW[ix].astype(np.float32)     # 按表现分功劳(启发式更新, 非单一损失的梯度)
        gsyn = gs.copy()                     # dL/d syn;辅助头那一路后面累加进来
        gSs_dir = 0.0                        # 辅助头把 Ss 当特征直接用 —— 这一路不带符号
        gsq_aux = 0.0
        gA = np.zeros_like(A); gc = np.zeros_like(c); gth = np.zeros_like(th)

        if BINS > 0:
            jL, fL, jR, fR = aux
            scatter_add(gth, jL,     g * (1 - fL)); scatter_add(gth, jL + 1,  g * fL)
            scatter_add(gth, jR,    -g * (1 - fR)); scatter_add(gth, jR + 1, -g * fR)

        if LAM > 0 and MARGIN is None:
            wsb = WS[ix] if WINCTRL else None
            feat = perf_feat(Ss, syn, sq, wsb)
            r = feat @ A + c - PERF[ix].astype(np.float32)
            if WITHIN: r = r - r.mean(1, keepdims=True)
            gp = (2.0 * LAM * r * (PW if PW is not None else 1.0) / (B * 10 * F)).astype(np.float32)
            gA += feat.reshape(-1, NF).T @ gp.reshape(-1, F)
            gc += gp.sum((0, 1))
            gf = gp @ A.T
            gSs_dir = gf[..., :K] + ts(gf[..., K:2 * K]) + tsw(gf[..., 2 * K:3 * K])
            if WINCTRL: gSs_dir = gSs_dir + wsb[..., None] * gf[..., 3 * K + 2:4 * K + 2]
            gsyn = gsyn + gf[..., 3 * K]     # 辅助头也经由 syn 影响嵌入
            gsq_aux = gf[..., 3 * K + 1][..., None]

        # syn = ½(Σ_k s_k·Ss_k² − Σ_i Σ_k s_k·e_ik²) —— 两条路都要逐维乘 s_k;
        # 辅助头直接用 Ss 的那一路不乘。SGN 全 1 时整段退化回原式。
        gsq = -.5 * gsyn[..., None] * np.ones((1, 1, 5), np.float32) + gsq_aux
        gSs = gSs_dir + gsyn[..., None] * Ss * SGN
        gE = gSs[:, :, None, :] + 2.0 * gsq[..., None] * E * SGN
        gw = np.zeros_like(w); scatter_add(gw, S.ravel(), np.repeat(gs, 5, axis=1).ravel())
        ge = np.zeros_like(e); scatter_add(ge, S.ravel(), gE.reshape(-1, K))
        gdw = np.zeros_like(dw)
        if PATCHSPLIT:
            gflat = np.zeros(NPATCH * D, np.float32)
            scatter_add(gflat, (pi[:, None, None] * D + S).ravel(), np.repeat(gs, 5, axis=1).ravel())
            gdw[:] = gflat.reshape(NPATCH, D); gdw[0] = 0

        t += 1
        for i, (p_, gr) in enumerate(zip(PS, (gw, ge, gA, gc, gth, gdw, gab))):
            lr_i = LR_TH if i == 4 else LR
            mo[i] = b1 * mo[i] + (1 - b1) * gr; ve[i] = b2 * ve[i] + (1 - b2) * gr * gr
            p_ -= lr_i * (mo[i] / (1 - b1 ** t)) / (np.sqrt(ve[i] / (1 - b2 ** t)) + eps)

    zv = np.concatenate([fwd(seats[va][i:i + 8192])[0] for i in range(0, va.stop - va.start, 8192)])
    vl = log_loss(y[va], np.clip(1 / (1 + np.exp(-zv)), 1e-6, 1 - 1e-6))
    if vl < best[0]: best = (vl, [p.copy() for p in PS], ep + 1)
    extra = ('  θ=' + ' '.join(f'{x:+.3f}' for x in th)) if BINS > 0 else ''
    if MARGIN is not None: extra += '  幅度相关 %.3f' % np.corrcoef(zv, MARGIN[va])[0, 1]
    print(f'  ep {ep+1:3d} 验证胜负 logloss {vl:.4f}  {time.time()-t0:.0f}s{extra}', flush=True)
    if ep + 1 - best[2] >= PAT:
        print(f'  连续 {PAT} 轮没进步,提前停在第 {ep+1} 轮(最优 {best[2]})', flush=True)
        break

w, e, A, c, th, dw, ab = best[1]
if MARGIN is not None: print('幅度头 α=%.3f β=%.3f' % (ab[0], ab[1]))
print(f'早停第 {best[2]} 轮')
r = bench.evaluate(TAG, score, note=f'K={K} λ={LAM} bins={BINS} lr={LR} ep={best[2]}/{EP}')
json.dump(r, open(f'result_{TAG}.json', 'w'))
np.savez_compressed(f'model_{TAG}.npz', w=w, e=e, A=A, c=c, th=th, fw=fw, sgn_dim=SGN, dw=dw,
                    knot=(KNOT if BINS > 0 else np.zeros(1)), ids=ids)
