import json, glob, sys, collections
"""吻合度/离谱率: 每个决策点以"标准答案"(全部候选×64, 前8再补512)为准。
   损失 = 标准答案第一名的胜率 − 该方法首推在标准答案里的胜率(pp)。
   安全阀 S(p,m): A 的首推如果网络概率 < p 且在 A 自己的打分里比网络首选高不到 m → 改推网络首选。"""
pts = [json.loads(l) for f in glob.glob(sys.argv[1]) for l in open(f) if l.startswith('{')]
print(f"决策点 {len(pts)} 个")
def loss(pt, key):
    ref = pt['ref']; best = max(v[0] for v in ref.values()); return 100 * (best - ref[key][0]) if key in ref else None
def safety(pt, p, m):
    rows = pt['Arows']; top = rows[0][0]; net = pt['net']; nt = max(net, key=net.get)
    if net.get(top, 0) >= p or top == nt: return top
    wA = dict((k, w) for k, w, n in rows)
    return nt if wA[top] - wA.get(nt, 0) < m else top
arms = collections.OrderedDict((a, (lambda a: lambda pt: pt['picks'][a])(a)) for a in ['N', 'A', 'B', 'C', 'Cw', 'F', 'F2'])
for p in [0.01, 0.05]:
    for m in [0.01, 0.02, 0.03, 0.05]:
        arms[f"A+阀(网<{int(p*100)}%,差<{int(m*100)})"] = (lambda p, m: lambda pt: safety(pt, p, m))(p, m)
print(f"{'方法':22}{'吻合':>6}{'差<1pp':>8}{'平均损失':>9}{'>3pp':>7}{'>5pp':>7}{'>10pp':>7}{'最大':>7}  改动")
for name, fn in arms.items():
    L = []; same = 0; close = 0; changed = 0
    for pt in pts:
        k = fn(pt); l = loss(pt, k)
        if l is None: continue
        L.append(l); best = max(pt['ref'], key=lambda x: pt['ref'][x][0]); same += k == best; close += l < 1
        changed += k != pt['picks']['A']
    n = len(L)
    print(f"{name:22}{100*same/n:5.0f}%{100*close/n:7.0f}%{sum(L)/n:8.2f}pp{100*sum(x>3 for x in L)/n:6.1f}%{100*sum(x>5 for x in L)/n:6.1f}%{100*sum(x>10 for x in L)/n:6.1f}%{max(L):6.1f}  {changed}")
ms = collections.defaultdict(list)
for pt in pts:
    for k, v in pt['ms'].items(): ms[k].append(v)
print("平均耗时(并行 6 进程下):", {k: int(sum(v) / len(v)) for k, v in ms.items()})
