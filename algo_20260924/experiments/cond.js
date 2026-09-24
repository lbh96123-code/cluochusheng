"use strict";
/* 条件推演: 某手强制某人拿 X, 推到终局 512 遍: 看某件东西最后归谁/第几手, 以及按"他拿到/没拿到"分组的胜率 */
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
const NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
const W = F.loadModel(__dirname + "/engine/public"), SC = W.ADScore, skey = k => { const h = W.AD_HEROES.find(x => x.key === k); return h ? "hero:" + h.id : k; };
async function cond(gfile, j, forceKey, trackKey, B = 512) {
  const g = JSON.parse(fs.readFileSync(__dirname + "/" + gfile)), pool = AR.poolOf(g), st = AR.stateAt(pool, g.seq, j), me = AR.FULL[j];
  const sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), fi = sim0.items.findIndex(x => x.key === forceKey), ti = sim0.items.findIndex(x => x.key === trackKey);
  const sims = [], owns = []; for (let b = 0; b < B; b++) { const s = F.cloneSim(sim0), o = Int8Array.from(own0); F.applySim(s, fi); o[fi] = me; sims.push(s); owns.push(o); }
  const at = new Int16Array(B).fill(-1); let k = 1;
  while (!F.simDone(sims[0])) { const ps = await C.netProbsBatch(sims, owns, sims.map(() => j + k)); const seat = sims[0].order[sims[0].step];
    for (let b = 0; b < B; b++) { let u = AR.mix(777 + b, k), pick = -1; for (let i = 0; i < 60; i++) { if (ps[b][i] <= 0) continue; u -= ps[b][i]; pick = i; if (u <= 0) break; }
      if (pick < 0) { F.passSim(sims[b]); continue; } F.applySim(sims[b], pick); owns[b][pick] = seat; if (pick === ti) at[b] = j + k + 1; } k++; }
  const sg = me < 5 ? 1 : -1, win = b => 1 / (1 + Math.exp(-sg * sims[b].z));
  const mine = [], other = [], who = {}; for (let b = 0; b < B; b++) { const o = owns[b][ti]; who[o < 0 ? "-" : NM[o]] = (who[o < 0 ? "-" : NM[o]] || 0) + 1; (o === me ? mine : other).push(b); }
  const avg = a => a.length ? (100 * a.reduce((s, b) => s + win(b), 0) / a.length).toFixed(1) : "-";
  const whenMine = mine.map(b => at[b]).sort((a, b) => a - b);
  console.log(`[${g.label || gfile} 第${j + 1}手 ${NM[me]} 强制拿 ${nm(forceKey)}] 全部平均胜率 ${avg([...Array(B).keys()])} | ${nm(trackKey)} 归属 ${JSON.stringify(who)} | 归我 ${mine.length}/${B} 中位第${whenMine[whenMine.length >> 1] || "-"}手, 归我时胜率 ${avg(mine)} / 没归我 ${avg(other)}`);
}
function pairGain(a, b) { const e = s => SC.evaluate(s).logit, z0 = e([[], [], [], [], [], [], [], [], [], []]);
  const one = k => e([[skey(k)], [], [], [], [], [], [], [], [], []]) - z0, both = e([[skey(a), skey(b)], [], [], [], [], [], [], [], [], []]) - z0;
  return { [nm(a)]: +one(a).toFixed(3), [nm(b)]: +one(b).toFixed(3), 两件一起: +both.toFixed(3), 配合项: +(both - one(a) - one(b)).toFixed(3) }; }
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  console.log("打分模型(同一座位, logit):", JSON.stringify(pairGain("batrider_sticky_napalm", "npc_dota_hero_ember_spirit")));
  await cond("batgame.json", 4, "batrider_sticky_napalm", "npc_dota_hero_ember_spirit");
  await cond("batgame.json", 4, "sven_great_cleave", "npc_dota_hero_ember_spirit");
  await cond("batgame.json", 4, "sven_great_cleave", "batrider_sticky_napalm");
})().catch(e => { console.error(e); process.exit(1); });
