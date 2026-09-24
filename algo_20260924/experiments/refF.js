"use strict";
/* F 调参: 重放 A 的对局(与 ref.js 同一批决策点), 每个点跑多组 F 参数, 输出首推; 与 r3 的标准答案离线对照 */
const AR = require("./arena.js"), { C, AI, F } = AR;
const [g0, g1] = (process.env.GAMES || "0-39").split("-").map(Number), [sh, nsh] = (process.env.SHARD || "0/1").split("/").map(Number);
const VARS = JSON.parse(process.env.VARS || '[{"c":0.05},{"c":0.1},{"c":0.2},{"c":0.1,"b":1536}]');
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  for (let g = g0; g <= g1; g++) { if (g % nsh !== sh) continue;
    const P = AR.POOLS[g % AR.POOLS.length], pool = AR.poolOf(P), seat = (g * 7 + 3) % 10, seq = [];
    for (let j = 0; j < AR.FULL.length; j++) {
      const who = AR.FULL[j], st = AR.stateAt(pool, seq, j), sim = AI._simFrom(st); if (F.simDone(sim)) break; let key;
      if (who === seat) { const seed = 100000000 + g * 7919 + j * 104729, out = {}, ms = {};
        C.LAST.screen = null; C.LAST.exact = null; key = (await C.refine(st, who, j, { seed })).rows[0].key;
        for (const v of VARS) { const q = Date.now(), r = await AR.decideF(st, who, j, seed, v.b || 768, { c: v.c, both: true, wave: v.w, minN: v.m, vl: v.vl }); const tag = `c${v.c}_b${v.b || 768}${v.w ? "_w" + v.w : ""}${v.vl ? "_" + v.vl : ""}`; out[tag + "_N"] = r.maxN; out[tag + "_Q"] = r.maxQ; ms[tag] = Date.now() - q; }
        process.stdout.write(JSON.stringify({ g, j, A: key, picks: out, ms }) + "\n");
      } else { const own = C.ownerOf(st, sim, who), p = await AR.netProbs(sim, own, j); let u = AR.mix(g + 1, j), pick = -1; for (let i = 0; i < 60; i++) { if (p[i] <= 0) continue; u -= p[i]; pick = i; if (u <= 0) break; } key = sim.items[pick].key; }
      seq.push(key);
    }
  }
})().catch(e => { console.error(e); process.exit(1); });
