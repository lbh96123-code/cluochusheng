"use strict";
/* 四个选技策略,全部跑在 fast.js 的增量引擎上。
   greedy   贪心:Δlogit 最大
   denial   贪心 + 一步封锁(线上现役):我方 top-K,各让对手贪心回一手,取回手后局面最好的
   gplay    贪心走子推演:我方 top-K,各把余下 50 手双方贪心走完,比终局分 —— 确定性,深度=全场
   mcts     根节点 top-K,每个候选跑 N 次 softmax(T) 走子到满编,取均值;所有候选共用同一批随机数 */
const F = require("./mcts_fast.js");

const BI = new Int32Array(640), BV = new Float64Array(640);
const CI = new Int32Array(640), CV = new Float64Array(640);
function topK(s, K, oi, ov) {
  const n = F.scoreAll(s, oi, ov);
  const ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => ov[b] - ov[a]);
  return ord.slice(0, Math.min(K, n)).map(j => ({ i: oi[j], v: ov[j] }));
}
function greedy(s) { const n = F.scoreAll(s, BI, BV); if (!n) return -1;
  let b = 0; for (let i = 1; i < n; i++) if (BV[i] > BV[b]) b = i; return BI[b]; }

function denial(s, K = 10) {
  const cands = topK(s, K, BI, BV); if (!cands.length) return -1;
  const sg = F.moverSign(s); let best = cands[0].i, bv = -Infinity;
  for (const c of cands) {
    const t = F.cloneSim(s); F.applySim(t, c.i);
    let v = sg * t.z;
    if (!F.simDone(t)) { const m = greedy(t); if (m >= 0) { F.applySim(t, m); v = sg * t.z; } }
    if (v > bv) { bv = v; best = c.i; }
  }
  return best;
}
function playoutGreedy(s) { const t = F.cloneSim(s); while (!F.simDone(t)) { const m = greedy(t); if (m < 0) break; F.applySim(t, m); } return t.z; }
function gplay(s, K = 10) {
  const cands = topK(s, K, BI, BV); if (!cands.length) return -1;
  const sg = F.moverSign(s); let best = cands[0].i, bv = -Infinity;
  for (const c of cands) { const t = F.cloneSim(s); F.applySim(t, c.i); const v = sg * playoutGreedy(t);
    if (v > bv) { bv = v; best = c.i; } }
  return best;
}

/* 推演时双方都用一步封锁(比贪心更接近真实对手),深度仍到满编 */
function playoutDenial(s, K) { const t = F.cloneSim(s);
  while (!F.simDone(t)) { const m = denial(t, K); if (m < 0) break; F.applySim(t, m); } return t.z; }
function gplayd(s, K = 10, KD = 6) {
  const cands = topK(s, K, BI, BV); if (!cands.length) return -1;
  const sg = F.moverSign(s); let best = cands[0].i, bv = -Infinity;
  for (const c of cands) { const t = F.cloneSim(s); F.applySim(t, c.i); const v = sg * playoutDenial(t, KD);
    if (v > bv) { bv = v; best = c.i; } }
  return best;
}
const mkRnd = seed => { let S = seed | 0; return () => { S |= 0; S = (S + 0x6D2B79F5) | 0;
  let t = Math.imul(S ^ (S >>> 15), 1 | S); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
/* softmax(T) 走子到满编 */
function playoutSoft(s, T, rnd) {
  const t = F.cloneSim(s);
  while (!F.simDone(t)) {
    const n = F.scoreAll(t, CI, CV); if (!n) break;
    let mx = CV[0]; for (let i = 1; i < n; i++) if (CV[i] > mx) mx = CV[i];
    let Z = 0; for (let i = 0; i < n; i++) { CV[i] = Math.exp((CV[i] - mx) / T); Z += CV[i]; }
    let r = rnd() * Z, pick = CI[n - 1];
    for (let i = 0; i < n; i++) { r -= CV[i]; if (r <= 0) { pick = CI[i]; break; } }
    F.applySim(t, pick);
  }
  return t.z;
}
/* 共用随机数 + 逐轮淘汰(successive halving) */
function mcts(s, opt) {
  const K = opt.K || 10, T = opt.T || 0.15, N = opt.N || 300, seed0 = opt.seed || 1;
  const cands = topK(s, K, BI, BV); if (!cands.length) return -1;
  const sg = F.moverSign(s);
  let live = cands.map(c => { const t = F.cloneSim(s); F.applySim(t, c.i); return { i: c.i, st: t, sum: 0, n: 0 }; });
  if (live.length === 1) return live[0].i;
  const rounds = opt.halving === false ? 1 : 3;
  let budget = N, done = 0;
  for (let r = 0; r < rounds; r++) {
    const per = Math.max(8, Math.round((budget * (r === rounds - 1 ? 1 : 0.4)) / 1));
    for (let j = 0; j < per; j++) {
      const sd = seed0 + (done + j) * 7919;                 // 同一轮所有候选共用这个种子
      for (const c of live) c.sum += sg * playoutSoft(c.st, T, mkRnd(sd)), c.n++;
    }
    done += per;
    if (r < rounds - 1 && live.length > 2) {
      live.sort((a, b) => b.sum / b.n - a.sum / a.n);
      live = live.slice(0, Math.max(2, Math.ceil(live.length / 2)));
      budget = Math.max(8, budget - per);
    }
  }
  live.sort((a, b) => b.sum / b.n - a.sum / a.n);
  return live[0].i;
}

/* League 版:rollout 时随机抽一对性格不同的选手把余下的手走完。
   同一轮所有候选共用同一个种子 → 也就共用同一对选手(共用随机数),比的是"同样的对手下谁更好"。 */
function mctsLeague(s, opt) {
  const RS = require("./roster.js");
  const K = opt.K || 10, N = opt.N || 128, seed0 = opt.seed || 1;
  const cands = topK(s, K, BI, BV); if (!cands.length) return -1;
  if (cands.length === 1) return cands[0].i;
  const sg = F.moverSign(s);
  let live = cands.map(c => { const t = F.cloneSim(s); F.applySim(t, c.i); return { i: c.i, st: t, sum: 0, n: 0 }; });
  let budget = N, done = 0;
  for (let r = 0; r < 3; r++) {
    const per = Math.max(8, Math.round(budget * (r === 2 ? 1 : 0.4)));
    for (let j = 0; j < per; j++) {
      const sd = seed0 + (done + j) * 7919;
      for (const c of live) { c.sum += sg * RS.playoutLeague(c.st, mkRnd(sd)); c.n++; }
    }
    done += per;
    if (r < 2 && live.length > 2) {
      live.sort((a, b) => b.sum / b.n - a.sum / a.n);
      live = live.slice(0, Math.max(2, Math.ceil(live.length / 2)));
      budget = Math.max(8, budget - per);
    }
  }
  live.sort((a, b) => b.sum / b.n - a.sum / a.n);
  return live[0].i;
}

module.exports = { greedy, denial, gplay, gplayd, playoutDenial, mctsLeague, mcts, playoutGreedy, playoutSoft, mkRnd, topK };
