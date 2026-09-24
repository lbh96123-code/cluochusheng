"use strict";
/* 配对单座位对战场。每一盘: 真实池子 + 指定座位 s 用"被测方法", 其余 9 人按网络概率抽样(每一步的随机数只由 盘号+手数 决定,
   各方法看到同一套随机数)。选满后用打分模型算 s 所在队的胜率。同一盘各方法轮流打一遍 → 配对比较。
   用法: ARMS=A,B,C,Cw GAMES=0-199 SHARD=0/4 node arena.js > out.jsonl */
const APP = __dirname, fs = require("fs");
const E = APP + "/engine/server"; const Dr = require(E + "/draft.js"), AI = require(E + "/ai.js"), F = require(E + "/mcts_fast.js");
const C = require(APP + "/exp_local.js");
const win = {}; new Function("window", fs.readFileSync(APP + "/engine/public/ad_data.js", "utf8"))(win); Dr.setExclusive(win.AD_EXCLUSIVE || []);
const POOLS = JSON.parse(fs.readFileSync(APP + "/real_pools.json"));
const FULL = (() => { const r = []; for (let i = 0; i < 5; i++) { r.push(i); r.push(5 + i); } const o = []; for (let k = 0; k < 5; k++) o.push(...(k % 2 ? r.slice().reverse() : r)); return o; })();
const ARMS = (process.env.ARMS || "A").split(","), [g0, g1] = (process.env.GAMES || "0-9").split("-").map(Number);
const [sh, nsh] = (process.env.SHARD || "0/1").split("/").map(Number);

function poolOf(P) { const pool = { heroKeys: P.heroes.slice(), basics: [], ults: [], filled: [] }; for (const s of P.skills) (s.ult ? pool.ults : pool.basics).push(s.key); return pool; }
function stateAt(pool, seq, d) { const st = Dr.newState(pool); const placed = [];
  for (let j = 0; j < d; j++) { const seat = st.seats[FULL[j]], k = seq[j]; if (k == null) continue;
    if (pool.heroKeys.includes(k)) seat.hero = k; else if (pool.ults.includes(k)) seat.ult = k; else seat.basics.push(k); seat.seq.push(k); placed.push(k); }
  st.taken = placed; st.blocked = []; st.order = FULL.slice(d); st.step = 0; return st; }
const side = s => s < 5 ? 0 : 1;
const mix = (a, b) => { let x = (Math.imul(a >>> 0, 0x9e3779b1) + b) >>> 0; x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0; x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0; return ((x ^ (x >>> 16)) >>> 0) / 4294967296; };
/* 单个局面的网络出手概率(合法件上 softmax) */
async function netProbs(sim, own, t) { const b = C.mkBuf(1); C.encodeRange([sim], [own], Math.min(49, t), b, 0); const L = await C.logits(b, 1);
  let mx = -Infinity; for (let i = 0; i < 60; i++) if (b.legal[i] && L[i] > mx) mx = L[i];
  const p = new Float64Array(60); let Z = 0; for (let i = 0; i < 60; i++) if (b.legal[i]) { p[i] = Math.exp(L[i] - mx); Z += p[i]; } for (let i = 0; i < 60; i++) p[i] /= Z; return p; }
const argmax = p => { let b = -1, v = -1; for (let i = 0; i < p.length; i++) if (p[i] > v) { v = p[i]; b = i; } return b; };

/* ---- 各方法: 返回要拿的 key ---- */
async function decideA(st, me, j, seed) { C.LAST.screen = null; C.LAST.exact = null; const r = await C.refine(st, me, j, { seed }); return r.rows[0].key; }
async function decideB(st, me, j, seed) { C.setTemp(+process.env.TEMP_B || 1.5); try { return await decideA(st, me, j, seed); } finally { C.setTemp(1); } }
/* C: 我的前 10 候选 × 下一个对手的 3 种应对(网络前 2 + 一步加分最高的 1 个), 各推 R 局; 返回 {avg, worst} */
const CR = +process.env.C_R || 24, CTOP = +process.env.C_TOP || 10;
async function decideC(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me);
  if (cands.length <= 1) return { avg: sim0.items[cands[0].i].key, worst: sim0.items[cands[0].i].key };
  const f1 = []; for (const c of cands) for (let r = 0; r < 8; r++) f1.push([[0, c.i]]);
  const r1 = await C.rolloutsG(sim0, own0, me, f1, seed, { t0: j });
  const m1 = cands.map((c, q) => { let s = 0; for (let r = 0; r < 8; r++) s += r1.v[q * 8 + r]; return { i: c.i, dep: c.dep, v: s / 8 }; }).sort((a, b) => (a.dep - b.dep) || (b.v - a.v));
  const top = m1.slice(0, CTOP);
  let k = 1; while (sim0.step + k < sim0.order.length && side(sim0.order[sim0.step + k]) === side(me)) k++;
  if (sim0.step + k >= sim0.order.length) { const key = sim0.items[top[0].i].key; return { avg: key, worst: key }; }
  const opp = sim0.order[sim0.step + k], sgO = opp < 5 ? 1 : -1;
  const plan = [];
  for (const c of top) {
    const s1 = F.cloneSim(sim0), o1 = Int8Array.from(own0); F.applySim(s1, c.i); o1[c.i] = me;
    for (let q = 1; q < k; q++) { const seat = s1.order[s1.step], p = await netProbs(s1, o1, j + q), a = argmax(p); F.applySim(s1, a); o1[a] = seat; }
    const p = await netProbs(s1, o1, j + k), legal = []; for (let i = 0; i < 60; i++) if (p[i] > 0 || (!s1.taken[i] && s1.items[i].p >= 0 && s1.slot[opp * 3 + s1.items[i].kind] < [1, 3, 1][s1.items[i].kind])) legal.push(i);
    const byNet = legal.slice().sort((a, b) => p[b] - p[a]), reps = byNet.slice(0, 2);
    let bs = -1, bv = -Infinity; for (const i of legal) { if (reps.includes(i)) continue; const v = sgO * F.deltaOf(s1, i); if (v > bv) { bv = v; bs = i; } } if (bs >= 0) reps.push(bs);
    plan.push({ c, reps: reps.map(r => ({ r, p: p[r] })) });
  }
  const f2 = [], idx = []; plan.forEach((pl, a) => pl.reps.forEach((rp, b) => { for (let r = 0; r < CR; r++) { f2.push([[0, pl.c.i], [k, rp.r]]); idx.push([a, b]); } }));
  const r2 = await C.rolloutsG(sim0, own0, me, f2, seed + 1, { t0: j });
  const sum = plan.map(pl => pl.reps.map(() => 0)); r2.v.forEach((v, q) => { const [a, b] = idx[q]; sum[a][b] += v / CR; });
  let bestA = -1, va = -1, bestW = -1, vw = -1;
  plan.forEach((pl, a) => { const w = pl.reps.map(rp => Math.max(rp.p, 0.1)), W = w.reduce((x, y) => x + y, 0);
    const avg = pl.reps.reduce((s, _, b) => s + w[b] / W * sum[a][b], 0), worst = Math.min(...sum[a]);
    if (avg > va) { va = avg; bestA = a; } if (worst > vw) { vw = worst; bestW = a; } });
  return { avg: sim0.items[plan[bestA].c.i].key, worst: sim0.items[plan[bestW].c.i].key };
}

/* F: 小 MCTS(PUCT)。值一律存"左队胜率", 选子时换成落子方视角。
   先验 = 网络概率 × (1-ε) + ε 平分给"这个落子方一步加分最高的 3 手"(根节点 ε 改为平分给全部合法件, 保证每个候选都会被看)。
   叶子 = 推演一局到满编(网络抽样)。每轮同时挑 WAVE 条路(虚拟损失分散), 按所在手分组批量推演。深度超过 DMAX 的节点不再展开。 */
const F_BUDGET = +process.env.F_BUDGET || 768, WAVE = +process.env.F_WAVE || 128, DMAX = +process.env.F_DMAX || 4, CPUCT = +process.env.F_C || 1.5;
const EPS = +process.env.F_EPS || 0.25, EPS_ROOT = +process.env.F_EPS_ROOT || 0.25, VLOSS = 1;
function mkNode(sim, own, depth) { return { sim, own, depth, mover: sim.order[sim.step], N: 0, W: 0, kids: null, term: F.simDone(sim) }; }
function priorsOf(node, pnet, root) {
  const s = node.sim, m = node.mover, sg = m < 5 ? 1 : -1, legal = [];
  for (let i = 0; i < 60; i++) if (pnet[i] > 0) legal.push(i);
  const P = new Float64Array(60); for (const i of legal) P[i] = pnet[i] * (1 - (root ? EPS_ROOT : EPS));
  if (root) for (const i of legal) P[i] += EPS_ROOT / legal.length;
  else { const g = legal.map(i => [i, sg * F.deltaOf(s, i)]).sort((a, b) => b[1] - a[1]).slice(0, 3); for (const [i] of g) P[i] += EPS / g.length; }
  node.kids = legal.map(i => ({ i, P: P[i], N: 0, W: 0, node: null }));
}
async function decideF(st, me, j, seed, budget = F_BUDGET, opt = {}) {
  const cp = opt.c != null ? opt.c : CPUCT, wv = opt.wave || WAVE, minQ = opt.minN || 16, vv = opt.vl === "visit";
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), root = mkNode(sim0, own0, 0);
  priorsOf(root, (await C.netProbsBatch([sim0], [own0], [j]))[0], true);
  if (root.kids.length === 1) return sim0.items[root.kids[0].i].key;
  let used = 0, wave = 0;
  while (used < budget) {
    const paths = [], leaves = [];
    for (let w = 0; w < wv && used + leaves.length < budget; w++) {
      let node = root; const path = [];
      while (true) {
        if (node.term || !node.kids) break;
        const lead = node.mover < 5, sqN = Math.sqrt(node.N + 1); let best = null, bv = -Infinity;
        const q0 = node.N ? (lead ? node.W / node.N : 1 - node.W / node.N) : 0.5;
        for (const e of node.kids) { const q = e.N ? (lead ? e.W / e.N : 1 - e.W / e.N) : q0; const u = q + cp * e.P * sqN / (1 + e.N); if (u > bv) { bv = u; best = e; } }
        /* 虚拟损失: loss = 当作落子方输了一局(胜率差 0.01~0.05 时会严重扭曲, 09-24 实测退化成照抄网络);
           visit = 当作按它当前胜率走了一次(只压低探索加分, 不改胜率) */
        const vw = vv ? (best.N ? best.W / best.N : (node.N ? node.W / node.N : 0.5)) : (lead ? 0 : VLOSS);
        path.push([node, best, vw]); best.N += VLOSS; best.W += vw;
        if (!best.node) { const s1 = F.cloneSim(node.sim), o1 = Int8Array.from(node.own); F.applySim(s1, best.i); o1[best.i] = node.mover; best.node = mkNode(s1, o1, node.depth + 1); node = best.node; break; }
        node = best.node; if (node.depth >= DMAX) break;
      }
      paths.push(path); leaves.push(node);
    }
    /* 新叶子(深度未到上限、非终局)补先验 —— 一次批量过网络 */
    const need = [...new Set(leaves.filter(n => !n.kids && !n.term && n.depth < DMAX))];
    if (need.length) { const ps = await C.netProbsBatch(need.map(n => n.sim), need.map(n => n.own), need.map(n => j + n.depth)); need.forEach((n, q) => priorsOf(n, ps[q], false)); }
    /* 叶子估值: 终局直接算, 否则按所在手分组推演一局 */
    const val = new Float64Array(leaves.length), groups = new Map();
    leaves.forEach((n, q) => { if (n.term) val[q] = 1 / (1 + Math.exp(-n.sim.z)); else { const k = n.sim.step; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(q); } });
    for (const [k, qs] of groups) { const v = await C.rolloutsMulti(qs.map(q => leaves[q].sim), qs.map(q => leaves[q].own), seed + wave * 7 + k, { t0: j + leaves[qs[0]].depth }); qs.forEach((q, a) => val[q] = v[a]); }
    /* 回传: 去掉虚拟损失, 加真实值(左队胜率) */
    paths.forEach((path, q) => { for (const [node, e, vw] of path) { e.N -= VLOSS; e.W -= vw;
      e.N += 1; e.W += val[q]; node.N += 1; node.W += val[q]; } const lf = leaves[q]; lf.N += 1; lf.W += val[q]; });
    used += leaves.length; wave++;
  }
  let best = null; for (const e of root.kids) if (!best || e.N > best.N || (e.N === best.N && e.W > best.W)) best = e;
  const lead = me < 5, q = e => lead ? e.W / e.N : 1 - e.W / e.N; let bq = null;
  for (const e of root.kids) if (e.N >= minQ && (!bq || q(e) > q(bq))) bq = e;
  if (opt.both) return { maxN: sim0.items[best.i].key, maxQ: sim0.items[(bq || best).i].key, visits: root.kids.filter(e => e.N > 0).length };
  return sim0.items[best.i].key;
}
async function decideN(st, me, j) { const sim = AI._simFrom(st), own = C.ownerOf(st, sim, me); return sim.items[argmax(await netProbs(sim, own, j))].key; }
const DECIDE = { Fv: (st, me, j, seed) => decideF(st, me, j, seed, 768, { c: 0.05, wave: 32, vl: "visit" }), F: decideF, F2: (st, me, j, seed) => decideF(st, me, j, seed, 2 * F_BUDGET), N: decideN, A: decideA, B: decideB, C: async (...a) => (await decideC(...a)).avg, Cw: async (...a) => (await decideC(...a)).worst };

async function play(g, arm) {
  const P = POOLS[g % POOLS.length], pool = poolOf(P), seat = (g * 7 + 3) % 10, seq = [], T0 = Date.now(); let tDec = 0, nDec = 0;
  for (let j = 0; j < FULL.length; j++) {
    const who = FULL[j], st = stateAt(pool, seq, j), sim = AI._simFrom(st);
    if (F.simDone(sim)) break;
    let key;
    if (who === seat) { const q = Date.now(); key = await DECIDE[arm](st, who, j, 100000000 + g * 7919 + j * 104729); tDec += Date.now() - q; nDec++; }
    else { const own = C.ownerOf(st, sim, who), p = await netProbs(sim, own, j); let u = mix(g + 1, j), pick = -1; for (let i = 0; i < 60; i++) { if (p[i] <= 0) continue; u -= p[i]; pick = i; if (u <= 0) break; } key = sim.items[pick].key; }
    seq.push(key);
  }
  const stF = stateAt(pool, seq, seq.length), simF = AI._simFrom(stF), zL = simF.z, wLeft = 1 / (1 + Math.exp(-zL));
  const short = stF.seats.filter(s => (s.hero ? 1 : 0) + s.basics.length + (s.ult ? 1 : 0) < 5).length;
  return { g, pool: g % POOLS.length, seat, arm, win: side(seat) === 0 ? wLeft : 1 - wLeft, mine: stF.seats[seat].seq, short, msDec: nDec ? Math.round(tDec / nDec) : 0, ms: Date.now() - T0 };
}
module.exports = { DECIDE, decideA, decideB, decideC, decideF, decideN, netProbs, stateAt, poolOf, POOLS, FULL, mix, side, argmax, C, AI, F };
if (require.main === module) (async () => { await C.init(APP + "/data/netB300.onnx", process.env.CPU ? "cpu" : "gpu", 128);
  for (let g = g0; g <= g1; g++) { if (g % nsh !== sh) continue;
    for (const arm of ARMS) { const r = await play(g, arm); process.stdout.write(JSON.stringify(r) + "\n"); } }
})().catch(e => { console.error(e); process.exit(1); });
