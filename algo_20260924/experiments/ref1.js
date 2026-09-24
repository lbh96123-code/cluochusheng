"use strict";
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
const g = JSON.parse(fs.readFileSync(__dirname + "/" + process.env.GAMEF)), pool = AR.poolOf(g), j = +process.env.J, me = AR.FULL[j];
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  const st = AR.stateAt(pool, g.seq, j), sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { const per = Math.max(1, Math.floor(8192 / R)); for (let a = 0; a < list.length; a += per) { const part = list.slice(a, a + per), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; }, se = i => { const a = S.get(i), m = mean(i); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length / a.length); };
  await run(cands, 128, 11); await run(cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, 12), 1024, 22);
  const own = C.ownerOf(st, sim0, me), np = await AR.netProbs(sim0, own, j);
  const rows = cands.map(c => [sim0.items[c.i].key, mean(c.i), se(c.i), S.get(c.i).length, np[c.i]]).sort((a, b) => b[1] - a[1]);
  rows.slice(0, 12).forEach((r, q) => console.log(`${q + 1}. ${nm(r[0])} ${(100 * r[1]).toFixed(1)} ±${(100 * r[2]).toFixed(1)} (${r[3]}局, 网络${(100 * r[4]).toFixed(1)}%)`));
  const ks = ["dazzle_shadow_wave", "tidehunter_anchor_smash", "npc_dota_hero_razor", "dark_seer_wall_of_replica", "npc_dota_hero_tusk"]; for (const k of ks) { const q = rows.findIndex(r => r[0] === k); if (q >= 0) console.log(`  ${nm(k)}: 第${q + 1} ${(100 * rows[q][1]).toFixed(1)}`); }
})().catch(e => { console.error(e); process.exit(1); });
