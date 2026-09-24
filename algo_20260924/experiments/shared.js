"use strict";
const AR = require("./arena.js"), { C, AI, F } = AR, fs = require("fs");
const meta = JSON.parse(fs.readFileSync(__dirname + "/meta.json")), nm = k => meta.cn.ab[k] || meta.cn.hero[k] || k;
const g = JSON.parse(fs.readFileSync(__dirname + "/optgame.json")), pool = AR.poolOf(g), j = 4, me = AR.FULL[j];
(async () => { await C.init(__dirname + "/data/netB300.onnx", "gpu", 128);
  const st = AR.stateAt(pool, g.seq, j), sim0 = AI._simFrom(st), own0 = C.ownerOf(st, sim0, me), cands = C.myCands(sim0, me).map(c => c.i);
  const I = +process.env.ITERS || 5, t0 = Date.now();
  const r = await C.rolloutsShared(sim0, own0, me, cands, { iters: I, R: +process.env.R || 16, R2: +process.env.R2 || 48, keep: +process.env.KEEP || 12, share: +process.env.SHARE || 6, mix: +process.env.MIX || 0, nmin: +process.env.NMIN || 3, t0: j, seed: 7, beta: +process.env.BETA || 20, eps: +process.env.EPS || 0.2 });
  const ki = k => sim0.items.findIndex(x => x.key === k), nap = ki("batrider_sticky_napalm"), fg = ki("ember_spirit_flame_guard");
  r.hist.forEach((h, it) => { const s = [...h].sort((a, b) => b[1] - a[1]), rk = i => s.findIndex(x => x[0] === i) + 1;
    console.log(`第${it + 1}轮: 前5 ${s.slice(0, 5).map(([i, v]) => nm(sim0.items[i].key) + " " + (100 * v).toFixed(1)).join(" / ")} | 叠油 第${rk(nap)} ${(100 * h.get(nap)).toFixed(1)} | 烈火罩 第${rk(fg)} ${(100 * h.get(fg)).toFixed(1)}`); });
  // 沙盘里: L3 拿了烈火罩之后, 下一手 R3 在这个局面拿了什么、各自对 R3 的平均结果
  const o1 = Int8Array.from(own0); o1[fg] = me; const st1 = r.T.get(C.ownKey(o1));
  if (st1) { const rows = [...st1].map(([a, e]) => [nm(sim0.items[a].key), e.N, e.W / e.N]).sort((a, b) => b[1] - a[1]);
    const tot = rows.reduce((x, y) => x + y[1], 0); console.log(`沙盘: L3 拿烈火罩后 R3 这一手(共 ${tot} 局): ` + rows.slice(0, 6).map(([n, N, q]) => `${n} ${(100 * N / tot).toFixed(0)}% (对R3 ${(100 * q).toFixed(1)})`).join(" | ")); }
  console.log(`用时 ${((Date.now() - t0) / 1000).toFixed(1)}s, 每轮 ${cands.length}×${+process.env.R || 16} 局`);
})().catch(e => { console.error(e); process.exit(1); });
