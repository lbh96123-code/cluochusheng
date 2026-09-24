import json,glob,sys,math,collections
rows=[json.loads(l) for f in sys.argv[1:] for g in [glob.glob(f)] for x in g for l in open(x) if l.strip().startswith('{')]
by=collections.defaultdict(dict)
for r in rows: by[r['g']][r['arm']]=r
arms=sorted({r['arm'] for r in rows}, key=lambda a:['N','A','B','C','Cw','F','F2'].index(a) if a in ['N','A','B','C','Cw','F','F2'] else 9)
print(f"盘数(各组都打完的): {sum(1 for g in by.values() if all(a in g for a in arms))}")
print(f"{'组':4}{'平均胜率':>8}{'每手耗时':>8}  vs A: 平均差(pp) ±误差  赢/平/输        vs N 平均差")
for a in arms:
    gs=[g for g in by.values() if a in g]
    w=sum(g[a]['win'] for g in gs)/len(gs); ms=sum(g[a]['msDec'] for g in gs)/len(gs); sh=sum(g[a]['short'] for g in gs)
    out=f"{a:4}{100*w:8.1f}{ms:8.0f}ms"
    for ref in ['A','N']:
        pr=[(g[a]['win']-g[ref]['win']) for g in gs if ref in g]
        if a==ref or not pr: out+="  "+" "*38; continue
        m=sum(pr)/len(pr); sd=math.sqrt(sum((x-m)**2 for x in pr)/max(1,len(pr)-1)); se=sd/math.sqrt(len(pr))
        wn=sum(x>1e-9 for x in pr); ls=sum(x<-1e-9 for x in pr); ti=len(pr)-wn-ls
        out+=f"  {100*m:+6.2f} ±{100*se:4.2f} (t{m/se if se>0 else 0:+5.1f}) {wn:3}/{ti:3}/{ls:3}" if ref=='A' else f"   {100*m:+6.2f} ±{100*se:4.2f}"
    if sh: out+=f"  [终局缺件座位 {sh}]"
    print(out)
