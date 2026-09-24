"use strict";
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
const W = F.loadModel(__dirname + "/engine/public"), SC = W.ADScore, sk = k => { const h = W.AD_HEROES.find(x => x.key === k); return h ? "hero:" + h.id : k; };
const g = JSON.parse(fs.readFileSync(__dirname + "/fdgame.json")), pool = AR.poolOf(g), j = 4, me = AR.FULL[j];
const NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const zSeat = arr => { const s = [[],[],[],[],[],[],[],[],[],[]]; s[0] = arr.map(sk); return SC.evaluate(s).logit; }, z0 = zSeat([]);
const one = k => zSeat([k]) - z0, pair = (a, b) => zSeat([a, b]) - z0 - one(a) - one(b);
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  const st = AR.stateAt(pool, g.seq, j), sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me);
  const cands = ["npc_dota_hero_tusk", "dark_seer_wall_of_replica", "dazzle_shadow_wave", "npc_dota_hero_razor"];
  console.log("单件分(logit):", cands.map(k => nm(k) + " " + one(k).toFixed(3)).join(" | "));
  console.log("一步加分(这个局面, 含和队友/对面的关系):", cands.map(k => nm(k) + " " + F.deltaOf(sim0, sim0.items.findIndex(x => x.key === k)).toFixed(3)).join(" | "));
  const all = sim0.items.map(x => x.key).filter(k => k !== "dazzle_shadow_wave");
  const pr = all.map(k => [k, pair("dazzle_shadow_wave", k)]).sort((a, b) => b[1] - a[1]);
  console.log("暗影波在同一座位的配合 前8:", pr.slice(0, 8).map(([k, v]) => nm(k) + " " + v.toFixed(2)).join(" | "));
  console.log("  最差3:", pr.slice(-3).map(([k, v]) => nm(k) + " " + v.toFixed(2)).join(" | "));
  const prT = all.map(k => [k, pair("npc_dota_hero_tusk", k)]).sort((a, b) => b[1] - a[1]);
  console.log("巨牙海民(巨牙海民)配合 前6:", prT.slice(0, 6).map(([k, v]) => nm(k) + " " + v.toFixed(2)).join(" | "));
  // 推演: 强制拿 X, L3 最后的阵容里最常见的件 & 胜率
  for (const x of ["dazzle_shadow_wave", "npc_dota_hero_tusk"]) {
    const xi = sim0.items.findIndex(t => t.key === x), f = []; for (let b = 0; b < 512; b++) f.push([[0, xi]]);
    const B = 512, sims = [], owns = []; for (let b = 0; b < B; b++) { const s = F.cloneSim(sim0), o = Int8Array.from(own0); F.applySim(s, xi); o[xi] = me; sims.push(s); owns.push(o); }
    let k = 1; while (!F.simDone(sims[0])) { const seat = sims[0].order[sims[0].step], ps = await C.netProbsBatch(sims, owns, sims.map(() => j + k));
      for (let b = 0; b < B; b++) { let u = AR.mix(900 + b, k), pick = -1; for (let i = 0; i < 60; i++) { if (ps[b][i] <= 0) continue; u -= ps[b][i]; pick = i; if (u <= 0) break; } if (pick < 0) { F.passSim(sims[b]); continue; } F.applySim(sims[b], pick); owns[b][pick] = seat; } k++; }
    const cnt = {}; let w = 0; for (let b = 0; b < B; b++) { w += 1 / (1 + Math.exp(-sims[b].z)); for (let i = 0; i < 60; i++) if (owns[b][i] === me && i !== xi) cnt[sim0.items[i].key] = (cnt[sim0.items[i].key] || 0) + 1; }
    console.log(`[L3 拿 ${nm(x)}] 左队胜率 ${(100 * w / B).toFixed(1)} | L3 后来最常拿: ` + Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k2, v]) => `${nm(k2)} ${(100 * v / B).toFixed(0)}%`).join(" / "));
  }
})().catch(e => { console.error(e); process.exit(1); });
