"use strict";
/* 实战决策点: 标准答案 vs 网络首选 vs 实际选择; 网络首选不是最优时, 看"最优那件"在拿了网络首选的推演里落到谁手、第几手被拿走 */
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const G = JSON.parse(fs.readFileSync(__dirname + "/real_games.json"));
const NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const SEATS = { "21:07 局": ["L2"], "21:58 局": ["L1"], "22:44 局": ["L5", "L1"], "00:01 局": ["L1"], "00:19 局": ["R1", "R2", "R3", "R4"] };
const [sh, nsh] = (process.env.SHARD || "0/1").split("/").map(Number);
async function reference(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { const per = Math.max(1, Math.floor(8192 / R)); for (let a = 0; a < list.length; a += per) { const part = list.slice(a, a + per), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  await run(cands, 64, seed); const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; };
  await run(cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, 8), 512, seed + 99991);
  const out = {}; for (const c of cands) out[sim0.items[c.i].key] = [+mean(c.i).toFixed(4), S.get(c.i).length]; return out;
}
/* 拿了 x 之后, y 的去向: 128 局推演里 y 最后归谁、平均第几手被拿 */
async function fate(st, me, j, xKey, yKey, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), xi = sim0.items.findIndex(t => t.key === xKey), yi = sim0.items.findIndex(t => t.key === yKey);
  const B = 128, sims = [], owns = []; for (let b = 0; b < B; b++) { const s = F.cloneSim(sim0), o = Int8Array.from(own0); F.applySim(s, xi); o[xi] = me; sims.push(s); owns.push(o); }
  const cnt = {}, when = []; let k = 1;
  while (!F.simDone(sims[0])) { const seat = sims[0].order[sims[0].step];
    const ps = await C.netProbsBatch(sims, owns, sims.map(() => j + k));
    for (let b = 0; b < B; b++) { let u = AR.mix(seed + b, k), pick = -1; for (let i = 0; i < 60; i++) { if (ps[b][i] <= 0) continue; u -= ps[b][i]; pick = i; if (u <= 0) break; }
      if (pick < 0) { F.passSim(sims[b]); continue; } F.applySim(sims[b], pick); owns[b][pick] = seat; if (pick === yi) when.push(k); }
    k++; }
  for (let b = 0; b < B; b++) { const o = owns[b][yi]; const who = o < 0 ? "没人" : o === me ? "我自己后来拿" : (AR.side(o) === AR.side(me) ? "队友" : "对面") + NM[o]; cnt[who] = (cnt[who] || 0) + 1; }
  return { cnt, avgWhen: when.length ? +(when.reduce((a, b) => a + b, 0) / when.length).toFixed(1) : null };
}
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128); let q = 0;
  for (const g of G) { const pool = AR.poolOf(g), seats = (SEATS[g.label] || []).map(n => NM.indexOf(n));
    for (let j = 0; j < 50; j++) { const me = AR.FULL[j]; if (!seats.includes(me)) continue; if (g.seq.slice(0, j + 1).some(x => x == null)) break; if (q++ % nsh !== sh) continue;
      const st = AR.stateAt(pool, g.seq, j), sim = AI._simFrom(st), own = C.ownerOf(st, sim, me), np = await AR.netProbs(sim, own, j);
      const netTop = sim.items[AR.argmax(np)].key, ref = await reference(st, me, j, 900000001 + j * 7919 + q);
      const best = Object.keys(ref).sort((a, b) => ref[b][0] - ref[a][0])[0], sg = me < 5 ? 1 : -1;
      const d1 = k => { const i = sim.items.findIndex(t => t.key === k); return +(sg * F.deltaOf(sim, i)).toFixed(3); };
      const rec = { game: g.label, j, seat: NM[me], actual: g.seq[j], netTop, netP: +np[sim.items.findIndex(t => t.key === netTop)].toFixed(3), best, ref, d1: { [netTop]: d1(netTop), [best]: d1(best) } };
      if (netTop !== best) rec.fate = await fate(st, me, j, netTop, best, 5000 + j);
      process.stdout.write(JSON.stringify(rec) + "\n"); } }
})().catch(e => { console.error(e); process.exit(1); });
