const AR = require("./arena.js"), C = require("./fatet/combo_local.js"), fs = require("fs");
const g = JSON.parse(fs.readFileSync(__dirname + "/fdgame.json")), NM = ["L1","L2","L3","L4","L5","R1","R2","R3","R4","R5"];
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128); const st = AR.stateAt(AR.poolOf(g), g.seq, 4), me = AR.FULL[4];
  for (const [a, b, an, bn] of [["dazzle_shadow_wave", "dark_seer_wall_of_replica", "暗影波", "复制之墙"], ["dark_seer_wall_of_replica", "dazzle_shadow_wave", "复制之墙", "暗影波"]]) {
    const f = await C.fate(st, me, 4, a, b, { seed: 5, B: 512, maxSteps: 20 });
    console.log(`L3 先拿${an}: ${bn} 到他下一手(${f.steps} 手内)被别人拿走 ${(100 * f.taken / f.B).toFixed(0)}%, 活下来 ${(100 - 100 * f.taken / f.B).toFixed(0)}% | 被谁: ${JSON.stringify(Object.fromEntries(Object.entries(f.bySeat).map(([k, v]) => [NM[k], v])))}`); }
})().catch(e => { console.error(e); process.exit(1); });
