const fs=require("fs"),D="/home/ec2-user/work/game/ad-draft/data/",A="/tmp/claude-1000/-home-ec2-user-work/6ab32d1d-17bb-423f-8103-c9df8b16460d/scratchpad/arena/";
const W=JSON.parse(fs.readFileSync(D+"wr_abilities.json")).data, ids=JSON.parse(fs.readFileSync(D+"dc_ability_ids.json")), meta=JSON.parse(fs.readFileSync(A+"meta.json"));
const F=require(A+"engine/server/mcts_fast.js"),M=F.loadModel(A+"engine/public"),SC=M.ADScore;
const heroKey=id=>{const h=M.AD_HEROES.find(x=>String(x.id)===String(id));return h?h.key:null;};
const keyOf=s=>s.abilityId<0?heroKey(-s.abilityId):ids[String(s.abilityId)];
const rows=W.abilityStats.map(s=>({key:keyOf(s),n:s.numPicks,pos:s.avgPickPosition,wr:s.winrate,val:W.abilityValuations[String(s.abilityId)]})).filter(r=>r.key&&r.n>=500);
const sk=k=>{const h=M.AD_HEROES.find(x=>x.key===k);return h?"hero:"+h.id:k;};
const z0=SC.evaluate([[],[],[],[],[],[],[],[],[],[]]).logit, one=k=>{const s=[[],[],[],[],[],[],[],[],[],[]];s[0]=[sk(k)];return SC.evaluate(s).logit-z0;};
rows.forEach(r=>r.m=one(r.key));
const N=rows.length, pct=(arr,v,hi)=>Math.round(100*arr.filter(x=>hi?x>v:x<v).length/N);
const WR=rows.map(r=>r.wr), POS=rows.map(r=>r.pos), MD=rows.map(r=>r.m);
const nm=k=>meta.cn.ab[k]||meta.cn.hero[k]||k;
console.log(`windrun 7.41d, 共 ${N} 件(被选≥500次)。"前x%"=在全部件里排前百分之几(胜率/单件分越高越前, 平均手数越小越前)`);
console.log("件 | 被选次数 | 胜率 | 平均第几手被抓 | 打分模型单件分(logit)");
for(const k of ["npc_dota_hero_tusk","dark_seer_wall_of_replica","dazzle_shadow_wave","npc_dota_hero_razor","batrider_sticky_napalm","ember_spirit_flame_guard","npc_dota_hero_ember_spirit"]){const r=rows.find(x=>x.key===k); if(!r){console.log(nm(k),"无数据");continue;}
 console.log(`${nm(k)} | ${r.n} | ${(100*r.wr).toFixed(1)}% (前${pct(WR,r.wr,true)}%) | ${r.pos.toFixed(1)} (前${pct(POS,r.pos,false)}%) | ${r.m.toFixed(3)} (前${pct(MD,r.m,true)}%)`);}
