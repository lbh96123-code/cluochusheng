"use strict";
/* 趣味测试: 每个玩家实际选择 vs 标准答案(全部候选×64 → 前8×512) */
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const G = JSON.parse(fs.readFileSync(__dirname + "/player_games.json"));
const NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const [sh, nsh] = (process.env.SHARD || "0/1").split("/").map(Number);
async function reference(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { const per = Math.max(1, Math.floor(8192 / R)); for (let a = 0; a < list.length; a += per) { const part = list.slice(a, a + per), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  await run(cands, +process.env.R1 || 64, seed); const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; };
  await run(cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, +process.env.TOPK || 8), +process.env.R2 || 512, seed + 99991);
  const out = {}; for (const c of cands) out[sim0.items[c.i].key] = +mean(c.i).toFixed(4); return out;
}
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128); let q = 0;
  for (let gi = 0; gi < G.length; gi++) { const g = G[gi], pool = AR.poolOf(g);
    for (const [iid, seatName] of Object.entries(g.players)) { const me = NM.indexOf(seatName); if (me < 0) continue;
      for (let j = 0; j < g.known; j++) { if (AR.FULL[j] !== me) continue; if (q++ % nsh !== sh) continue;
        const st = AR.stateAt(pool, g.seq, j), sim = AI._simFrom(st); if (F.simDone(sim)) continue;
        const own = C.ownerOf(st, sim, me), np = await AR.netProbs(sim, own, j), netTop = sim.items[AR.argmax(np)].key;
        const ref = await reference(st, me, j, 300000007 + gi * 7919 + j * 104729), best = Object.keys(ref).sort((a, b) => ref[b] - ref[a])[0];
        const act = g.seq[j];
        process.stdout.write(JSON.stringify({ gi, day: g.day, iid, seat: seatName, j, actual: act, inRef: act in ref, best, vBest: ref[best], vAct: ref[act], netTop, vNet: ref[netTop], n: Object.keys(ref).length }) + "\n"); } } }
})().catch(e => { console.error(e); process.exit(1); });
