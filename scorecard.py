#!/usr/bin/env python3
"""把 checkup_*.json / consensus_pf.json / eval_pf.log 汇成一张记分卡。用法: scorecard.py 基线tag tag1 tag2 ..."""
import json, os, re, sys
os.chdir('/root/adres/research')
base = sys.argv[1]; tags = sys.argv[2:]
cons = json.load(open('pairaudit/consensus_pf.json')) if os.path.exists('pairaudit/consensus_pf.json') else {}
ev = {}
for l in open('eval_pf.log') if os.path.exists('eval_pf.log') else []:
    m = re.match(r'(\S+)\s+logloss ([\d.]+) \| 稀有正配合该乘 ([\d.]+)/([\d.]+)\s+稀有负 ([\d.]+)/([\d.]+) \| 收缩 g\(n\) 再拿 Δ([+-][\d.]+)', l)
    if m: ev[m.group(1)] = dict(ll=float(m.group(2)), beta=(float(m.group(3)) + float(m.group(4))) / 2, gshr=float(m.group(7)))
def row(tag, R=None, key='new'):
    r = {}
    if R:
        r['ll'] = R['total'][key]; r['comp_rms'] = 100 * R['comp_rms_' + key]; r['comp_flags'] = R['comp_flags_' + key]
        r['calib_max'] = 100 * max(abs(v) for v in R['calib'][key])
        if 'rare' in R:
            for x in R['rare']:
                if x['lo'] == 1 and x['hi'] == 30: r['agree_1_30'] = x[key]['agree']; r['keep_1_30'] = x[key]['keep']
                if x['lo'] == 30 and x['hi'] == 100: r['agree_30_100'] = x[key]['agree']; r['keep_30_100'] = x[key]['keep']
                if x['lo'] == 100 and x['hi'] == 300: r['agree_100_300'] = x[key]['agree']; r['keep_100_300'] = x[key]['keep']
        if 'perf' in R: r['perf_mean'] = sum(R['perf'][key]) / len(R['perf'][key])
    if tag in ev: r['beta'] = ev[tag]['beta']; r['gshr'] = ev[tag]['gshr']
    if tag in cons: r['cons_r'] = cons[tag]['pearson']; r['cons_rho'] = cons[tag]['spearman']; r['cons_low'] = cons[tag]['low_rated_mean']
    return r
rows = {}
for t in tags:
    p = f'pairaudit/checkup_{t}.json'
    if os.path.exists(p):
        R = json.load(open(p)); rows[t] = row(t, R, 'new')
        if base not in rows: rows[base] = row(base, R, 'old')
    else: rows[t] = row(t)
cols = [('ll', '总误差', '%.5f'), ('comp_rms', '分组偏差pp', '%.2f'), ('comp_flags', '⚠组', '%d'), ('calib_max', '刻度最大pp', '%.2f'),
        ('agree_1_30', '一致1-30', '%.3f'), ('keep_1_30', '兑现1-30', '%.0f%%'), ('agree_30_100', '一致30-100', '%.3f'), ('keep_30_100', '兑现30-100', '%.0f%%'),
        ('agree_100_300', '一致100-300', '%.3f'), ('keep_100_300', '兑现100-300', '%.0f%%'), ('beta', '稀有正β', '%.2f'), ('perf_mean', '表现相关均', '%.3f'), ('cons_r', '人工相关', '%.3f'), ('cons_low', '人打≤2均pp', '%+.2f')]
print('%-46s' % 'tag' + ''.join('%12s' % c[1] for c in cols))
for t in [base] + [x for x in tags if x != base]:
    r = rows.get(t, {})
    def f(c):
        v = r.get(c[0]);
        if v is None: return '-'
        return c[2] % (v * 100 if c[2].endswith('%%') else v)
    print('%-46s' % t + ''.join('%12s' % f(c) for c in cols))
