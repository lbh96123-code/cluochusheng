"use strict";
/* AD 选技核心 —— 纯逻辑,不碰网络,可单测。
 *
 * 规则:
 *   · 开 12 英雄 → 池子 = 12 英雄 + 36 普通技能 + 12 大招 = 60 项
 *   · 大招被 AD 禁用的英雄(拉比克/米波/变体/天穹/酒仙/龙骑/食人魔/祈求者)
 *     和普通技能不足的,**随机从别的英雄补进来**
 *   · 5v5:两名玩家各代理 5 个座位,共 10 座
 *   · 每座要凑齐 1 英雄 + 3 普通 + 1 大招 = 5 手,全场 50 手
 *   · 每一手可以自由选英雄或技能,只要该座位那个槽还空着
 *   · 座位级蛇形,一人一手交替:
 *       正向  左1 右1 左2 右2 左3 右3 左4 右4 左5 右5
 *       反向  右5 左5 右4 左4 右3 左3 右2 左2 右1 左1
 *     五轮 正-反-正-反-正,共 50 手。
 *     轮次交界处(右5 → 右5)同一座位连选两手,这是蛇形固有的形态,不是 bug。
 */
const HERO_COUNT = 12, PER_SIDE = 5, BASICS_PER_SEAT = 3, PICKS_PER_SEAT = 5;

const shuffle = (a, rnd) => {
  a = a.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};

/* 同一技能槽的两个形态(Kez 刀/双叉、杰奇洛液火/液冰),池子里只会出现一个。
   由真实对局判出(tools/derive-ults.py)写进 ad_data.js 的 AD_EXCLUSIVE,启动时灌进来。 */
let EXCL = [];
const setExclusive = list => { EXCL = list || []; };
function buildPool(heroes, rnd = Math.random, excl = EXCL) {
  const picked = shuffle(heroes, rnd).slice(0, HERO_COUNT);
  const partner = new Map();
  for (const [a, b] of excl) { partner.set(a, b); partner.set(b, a); }
  /* 每组互斥形态随机留一个 */
  const collapse = list => {
    const drop = new Set();
    for (const a of list) {
      const b = partner.get(a);
      if (b && list.includes(b) && !drop.has(a) && !drop.has(b)) drop.add(rnd() < 0.5 ? a : b);
    }
    return list.filter(a => !drop.has(a));
  };
  const allB = heroes.flatMap(h => h.basics), allU = heroes.filter(h => h.ult).map(h => h.ult);
  const used = new Set(), basics = [], ults = [], filled = [];
  for (const h of picked) {
    /* 随机取 3 个,不是取前 3 个 —— 取前 3 会让技能多于 3 个的英雄(祈求者 10 个、
       Kez 6 个、杰奇洛/巨魔 4 个)永远只出前面那几个 */
    const free = a => !used.has(a) && !(partner.has(a) && used.has(partner.get(a)));
    const mine = shuffle(collapse(h.basics), rnd).filter(free).slice(0, BASICS_PER_SEAT);
    mine.forEach(a => used.add(a));
    while (mine.length < BASICS_PER_SEAT) {
      const c = allB[(rnd() * allB.length) | 0];
      if (used.has(c)) continue;
      if (partner.has(c) && used.has(partner.get(c))) continue;   // 另一个形态已在池里
      used.add(c); mine.push(c);
      filled.push({ key: c, why: `${h.name} 可选普通技能不足` });
    }
    basics.push(...mine);
    let u = h.ult;
    const up = u && partner.get(u);                 // 大招也可能有另一个形态
    if (up && rnd() < 0.5 && free(up)) u = up;
    if (!u || !free(u)) {
      do { u = allU[(rnd() * allU.length) | 0]; }
      while (used.has(u) || (partner.has(u) && used.has(partner.get(u))));
      filled.push({ key: u, why: h.ult ? `${h.name} 大招重复` : `${h.name} 的大招在 AD 中被禁用` });
    }
    used.add(u); ults.push(u);
  }
  return { heroKeys: picked.map(h => h.key), basics, ults, filled };
}

/* 一轮 10 手 = 左1 右1 左2 右2 … 左5 右5,下一轮整个倒过来,五轮共 50 手 */
function draftOrder() {
  const round = [];
  for (let i = 0; i < PER_SIDE; i++) { round.push(i); round.push(PER_SIDE + i); }
  const order = [];
  for (let r = 0; r < PICKS_PER_SEAT; r++) order.push(...(r % 2 ? round.slice().reverse() : round));
  return order;
}

function newState(pool) {
  return {
    pool,
    seats: Array.from({ length: PER_SIDE * 2 }, (_, i) => ({
      idx: i, side: i < PER_SIDE ? "L" : "R", hero: null, basics: [], ult: null,
      seq: [],                                  // 这个座位 5 次选择的先后顺序,前端标 1~5
    })),
    order: draftOrder(), step: 0, taken: [],
  };
}

const done = s => s.step >= s.order.length;
const curSeat = s => done(s) ? null : s.seats[s.order[s.step]];

/* 这个座位还缺哪几类 */
function slotsLeft(seat) {
  return {
    hero: !seat.hero,
    basic: seat.basics.length < BASICS_PER_SEAT,
    ult: !seat.ult,
  };
}

/* 选一项(英雄或技能)。key 是英雄 key 或技能 key */
function pick(s, key) {
  if (done(s)) return { err: "这一局已经选完了" };
  const seat = curSeat(s);
  const taken = new Set(s.taken);
  if (taken.has(key)) return { err: "已经被选走了" };

  let kind = null;
  if (s.pool.heroKeys.includes(key)) kind = "hero";
  else if (s.pool.ults.includes(key)) kind = "ult";
  else if (s.pool.basics.includes(key)) kind = "basic";
  else return { err: "不在本局池子里" };

  const left = slotsLeft(seat);
  if (!left[kind]) {
    return { err: { hero: "这个位置已经有英雄了", ult: "这个位置的大招已经选了", basic: "这个位置的普通技能满了" }[kind] };
  }
  /* 剩几手就必须开始补缺口:防止最后一手没法凑齐 */
  const remain = PICKS_PER_SEAT - (seat.hero ? 1 : 0) - seat.basics.length - (seat.ult ? 1 : 0);
  const needCount = (left.hero ? 1 : 0) + (left.ult ? 1 : 0) + (left.basic ? BASICS_PER_SEAT - seat.basics.length : 0);
  if (remain === needCount && kind === "basic" && seat.basics.length >= BASICS_PER_SEAT) {
    return { err: "剩下的手数不够补齐其他槽位了" };
  }

  if (kind === "hero") seat.hero = key;
  else if (kind === "ult") seat.ult = key;
  else seat.basics.push(key);
  seat.seq.push(key);                           // 记下这一手是这个座位的第几次选择
  s.taken.push(key);
  s.step++;
  return { ok: true, kind, seat: seat.idx };
}

/* 超时/托管时自动选一个合法的 */
function autoPick(s, rnd = Math.random) {
  const seat = curSeat(s);
  if (!seat) return null;
  const taken = new Set(s.taken), left = slotsLeft(seat);
  const remain = PICKS_PER_SEAT - (seat.hero ? 1 : 0) - seat.basics.length - (seat.ult ? 1 : 0);
  const needCount = (left.hero ? 1 : 0) + (left.ult ? 1 : 0) + (BASICS_PER_SEAT - seat.basics.length);
  /* 手数刚好够补缺口时,优先补最紧的槽 */
  const pools = [];
  if (left.hero) pools.push(s.pool.heroKeys);
  if (left.ult) pools.push(s.pool.ults);
  if (left.basic) pools.push(s.pool.basics);
  if (remain <= needCount) {
    if (left.hero) pools.unshift(s.pool.heroKeys);
    if (left.ult) pools.unshift(s.pool.ults);
  }
  for (const p of pools) {
    const c = p.filter(k => !taken.has(k));
    if (c.length) return c[(rnd() * c.length) | 0];
  }
  return null;
}

module.exports = { buildPool, setExclusive, draftOrder, newState, pick, autoPick, done, curSeat, slotsLeft,
  HERO_COUNT, PER_SIDE, BASICS_PER_SEAT, PICKS_PER_SEAT };
