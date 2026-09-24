"use strict";
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const g = JSON.parse(fs.readFileSync(__dirname + "/" + (process.env.GAMEF || "napgame.json"))), pool = AR.poolOf(g), NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
async function reference(st, me, j, seed) {
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me), S = new Map(cands.map(c => [c.i, []]));
  const run = async (list, R, sd) => { const per = Math.max(1, Math.floor(8192 / R)); for (let a = 0; a < list.length; a += per) { const part = list.slice(a, a + per), f = [];
      for (const c of part) for (let r = 0; r < R; r++) f.push([[0, c.i]]); const res = await C.rolloutsG(sim0, own0, me, f, sd + a, { t0: j });
      part.forEach((c, q) => { for (let r = 0; r < R; r++) S.get(c.i).push(res.v[q * R + r]); }); } };
  await run(cands, 64, seed); const mean = i => { const a = S.get(i); return a.reduce((x, y) => x + y, 0) / a.length; };
  await run(cands.slice().sort((a, b) => mean(b.i) - mean(a.i)).slice(0, 8), 512, seed + 99991);
  return cands.map(c => [sim0.items[c.i].key, mean(c.i)]).sort((a, b) => b[1] - a[1]);
}
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  // 1) 从第 1 手全场网络推演 512 遍: 叠油第几手、谁拿; 同时看凤凰本体/超新星
  const st0 = AR.stateAt(pool, g.seq, 0), sim0 = AI._simFrom(st0), own0 = C.ownerOf(st0, sim0, 0), B = 512;
  const track = ["batrider_sticky_napalm", "npc_dota_hero_ember_spirit", "npc_dota_hero_batrider", "ember_spirit_fire_remnant"], ti = track.map(k => sim0.items.findIndex(x => x.key === k));
  const sims = [], owns = []; for (let b = 0; b < B; b++) { sims.push(F.cloneSim(sim0)); owns.push(Int8Array.from(own0)); }
  const when = track.map(() => []), who = track.map(() => ({})); let k = 0;
  while (!F.simDone(sims[0])) { const seat = sims[0].order[sims[0].step]; const ps = await C.netProbsBatch(sims, owns, sims.map(() => k));
    for (let b = 0; b < B; b++) { let u = AR.mix(4242 + b, k), pick = -1; for (let i = 0; i < 60; i++) { if (ps[b][i] <= 0) continue; u -= ps[b][i]; pick = i; if (u <= 0) break; }
      if (pick < 0) { F.passSim(sims[b]); continue; } F.applySim(sims[b], pick); owns[b][pick] = seat; const q = ti.indexOf(pick); if (q >= 0) { when[q].push(k + 1); who[q][NM[seat]] = (who[q][NM[seat]] || 0) + 1; } }
    k++; }
  const napWho = new Array(B).fill(-1); for (let b = 0; b < B; b++) napWho[b] = owns[b][ti[0]];
  let same = 0; for (let b = 0; b < B; b++) if (owns[b][ti[0]] >= 0 && owns[b][ti[0]] === owns[b][ti[1]]) same++;
  track.forEach((t, q) => { const w = when[q].sort((a, b) => a - b); console.log(`[推演] ${nm(t)}: 平均第 ${(w.reduce((a, b) => a + b, 0) / w.length).toFixed(1)} 手被拿, 中位第 ${w[w.length >> 1]} 手, 最早 ${w[0]} 最晚 ${w[w.length - 1]} | 谁拿 ${JSON.stringify(who[q])}`); });
  console.log(`[推演] 叠油和凤凰本体落在同一个人手里: ${(100 * same / B).toFixed(0)}%`);
  // 2) 每一手叠油还在时, 当手选人的网络概率/标准答案里叠油排第几 (前 12 手); 3) 你的 5 手
  const napIdx = sim0.items.findIndex(x => x.key === "batrider_sticky_napalm");
  for (let j = 0; j < 50; j++) { const me = AR.FULL[j], mine = NM[me] === (process.env.ME || "L3");
    const st = AR.stateAt(pool, g.seq, j), sim = AI._simFrom(st); if (sim.taken[napIdx] && !mine) continue; if (j > 12 && !mine) continue;
    const own = C.ownerOf(st, sim, me), np = await AR.netProbs(sim, own, j), ref = await reference(st, me, j, 5100 + j), best = ref[0];
    const rk = ref.findIndex(x => x[0] === "batrider_sticky_napalm"), act = g.seq[j], ra = ref.findIndex(x => x[0] === act);
    console.log(`第${j + 1}手 ${NM[me]}${mine ? "(你)" : ""}: 实际 ${nm(act)}(标答第${ra + 1}, 少${(100 * (best[1] - ref[ra][1])).toFixed(1)}) | 标答前三 ${ref.slice(0, 3).map(x => nm(x[0]) + " " + (100 * x[1]).toFixed(1)).join(" / ")}` +
      (rk >= 0 ? ` | 叠油: 标答第${rk + 1} ${(100 * ref[rk][1]).toFixed(1)}, 网络${(100 * np[napIdx]).toFixed(1)}%` : " | 叠油已被拿"));
  }
})().catch(e => { console.error(e); process.exit(1); });
