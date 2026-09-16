#!/usr/bin/env python3
"""座位分(胜负模型 单件+座位内配合) 与 该座位实际表现 的相关(测试段). 用法: seatcorr.py tag1 tag2 ..."""
import numpy as np, os, sys, json
os.chdir('/root/adres/research')
d = np.load('data_pure_d.npz'); S = d['seats'].astype(np.int32); n = len(S); TR, VA = int(d['ntr']), int(d['nva'])
raw = np.load('data_perf_raw_d.npz'); fl = [str(x) for x in raw['fields']]; R = raw['raw']; mins = (raw['dur'] / 60.0).astype(np.float32)[:, None]
g = lambda f: R[:, :, fl.index(f)].astype(np.float32)
Y = np.stack([g('gpm'), g('xpm'), g('lastHits') / mins, g('heroDamage') / mins, np.log1p(g('heroHealing') / mins), g('kills') / mins, g('deaths') / mins, g('assists') / mins], 2); del R, raw
Z = (Y - Y[:TR].reshape(-1, 8).mean(0)) / (Y[:TR].reshape(-1, 8).std(0) + 1e-9); del Y
w8 = np.array(json.load(open('perfabs/quick_kda.json'))['w'], np.float32); comp = (Z[VA:] @ w8).ravel(); Zt = Z[VA:]
NM = ['gpm', 'xpm', '补刀', '伤害', '治疗', '击杀', '死亡', '助攻']
print('%-44s %8s | %s' % ('模型', '综合分', ' '.join('%6s' % k for k in NM)))
for tag in sys.argv[1:]:
    M = np.load(f'model_{tag}.npz'); w, e, SG = M['w'].astype(np.float32), M['e'].astype(np.float32), M['sgn_dim'].astype(np.float32)
    so = np.concatenate([(lambda E: w[Sb].sum(2) + .5 * (((E.sum(2)) ** 2 * SG).sum(2) - (E ** 2 * SG).sum(3).sum(2)))(e[Sb]) for Sb in (S[s:min(s + 4096, n)] for s in range(VA, n, 4096))]).ravel()
    print('%-44s %8.3f | %s' % (tag, np.corrcoef(so, comp)[0, 1], ' '.join('%6.3f' % np.corrcoef(so, Zt[:, :, j].ravel())[0, 1] for j in range(8))))
