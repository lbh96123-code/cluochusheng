"use strict";
/* 两边都按标准答案第一名选: 每手 全部候选×64 → 前8×512, 取第一名落子, 再算下一手 */
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
const NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const g = JSON.parse(fs.readFileSync(__dirname + "/" + (process.env.GAMEF || "batgame.json"))), pool = AR.poolOf(g), N = +process.env.N || 10;
async function reference(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { const per = Math.max(1, Math.floor(8192 / R)); for (let a = 0; a < list.length; a += per) { const part = list.slice(a, a + per), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  await run(cands, 64, seed); const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; };
  await run(cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, 8), 512, seed + 99991);
  return cands.map(c => [sim0.items[c.i].key, mean(c.i)]).sort((a, b) => b[1] - a[1]);
}
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128); const seq = [];
  for (let j = 0; j < N; j++) { const me = AR.FULL[j], st = AR.stateAt(pool, seq, j), ref = await reference(st, me, j, 6100 + j * 7919);
    const rk = ref.findIndex(x => x[0] === "batrider_sticky_napalm");
    console.log(`第${j + 1}手 ${NM[me]}: ${nm(ref[0][0])} ${(100 * ref[0][1]).toFixed(1)}  (第2 ${nm(ref[1][0])} ${(100 * ref[1][1]).toFixed(1)}, 第3 ${nm(ref[2][0])} ${(100 * ref[2][1]).toFixed(1)})` + (rk >= 0 ? `  | 叠油 第${rk + 1} ${(100 * ref[rk][1]).toFixed(1)}` : "  | 叠油已被拿"));
    seq.push(ref[0][0]); }
})().catch(e => { console.error(e); process.exit(1); });
