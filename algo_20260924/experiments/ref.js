"use strict";
/* 吻合度/离谱率评测: 重放 A 组对局, 每个被测座位的决策点上 ① 算"标准答案"(全部候选×64 → 前 8 再补 512)
   ② 各方法在同一局面给首推 ③ 存 A 的全部打分 + 网络概率(安全阀门槛离线调)。然后按 A 的选择继续。 */
const AR = require("./arena.js"), { C, AI, F } = AR;
const [g0, g1] = (process.env.GAMES || "0-0").split("-").map(Number), [sh, nsh] = (process.env.SHARD || "0/1").split("/").map(Number);
const R1 = +process.env.REF_R1 || 64, TOPK = +process.env.REF_TOP || 8, R2 = +process.env.REF_R2 || 512;
async function reference(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { for (let a = 0; a < list.length; a += Math.max(1, Math.floor(8192 / R))) { const part = list.slice(a, a + Math.max(1, Math.floor(8192 / R))), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  await run(cands, R1, seed);
  const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; };
  const top = cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, TOPK); await run(top, R2, seed + 99991);
  const out = {}; for (const c of cands) { const a = S.get(c.i), m = mean(c.i), sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); out[sim0.items[c.i].key] = [+m.toFixed(5), a.length, +(sd / Math.sqrt(a.length)).toFixed(5)]; }
  return out;
}
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  for (let g = g0; g <= g1; g++) { if (g % nsh !== sh) continue;
    const P = AR.POOLS[g % AR.POOLS.length], pool = AR.poolOf(P), seat = (g * 7 + 3) % 10, seq = [];
    for (let j = 0; j < AR.FULL.length; j++) {
      const who = AR.FULL[j], st = AR.stateAt(pool, seq, j), sim = AI._simFrom(st); if (F.simDone(sim)) break; let key;
      if (who === seat) {
        const seed = 100000000 + g * 7919 + j * 104729, T = {}, t = async (k, fn) => { const q = Date.now(); const r = await fn(); T[k] = Date.now() - q; return r; };
        C.LAST.screen = null; C.LAST.exact = null;
        const rA = await t("A", () => C.refine(st, who, j, { seed }));
        const own = C.ownerOf(st, sim, who), np = await AR.netProbs(sim, own, j), net = {}; for (let i = 0; i < 60; i++) if (np[i] > 0) net[sim.items[i].key] = +np[i].toFixed(5);
        const picks = { A: rA.rows[0].key, N: sim.items[AR.argmax(np)].key };
        picks.B = await t("B", () => AR.decideB(st, who, j, seed));
        const c = await t("C", () => AR.decideC(st, who, j, seed)); picks.C = c.avg; picks.Cw = c.worst;
        picks.F = await t("F", () => AR.decideF(st, who, j, seed));
        picks.F2 = await t("F2", () => AR.decideF(st, who, j, seed, 1536));
        const ref = await t("ref", () => reference(st, who, j, 700000001 + g * 7919 + j * 104729));
        process.stdout.write(JSON.stringify({ g, j, seat, picks, ms: T, Arows: rA.rows.map(r => [r.key, +r.win.toFixed(5), r.n]), net, ref }) + "\n");
        key = picks.A;
      } else { const own = C.ownerOf(st, sim, who), p = await AR.netProbs(sim, own, j); let u = AR.mix(g + 1, j), pick = -1; for (let i = 0; i < 60; i++) { if (p[i] <= 0) continue; u -= p[i]; pick = i; if (u <= 0) break; } key = sim.items[pick].key; }
      seq.push(key);
    }
  }
})().catch(e => { console.error(e); process.exit(1); });
