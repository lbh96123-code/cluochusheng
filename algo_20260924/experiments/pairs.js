const fs=require("fs"),F=require("./engine/server/mcts_fast.js"),W=F.loadModel("./engine/public"),SC=W.ADScore,meta=JSON.parse(fs.readFileSync("meta.json"));
const nm=k=>meta.cn.ab[k]||meta.cn.hero[k]||k, sk=k=>{const h=W.AD_HEROES.find(x=>x.key===k);return h?"hero:"+h.id:k;};
const z=seat=>{const s=[[],[],[],[],[],[],[],[],[],[]];s[0]=seat.map(sk);return SC.evaluate(s).logit;}, z0=z([]);
const one=k=>z([k])-z0, pair=(a,b)=>z([a,b])-z0-one(a)-one(b), pp=x=>(100/(1+Math.exp(-x))-50).toFixed(1);
const K=["batrider_sticky_napalm","ember_spirit_flame_guard","npc_dota_hero_ember_spirit","ember_spirit_fire_remnant","ember_spirit_searing_chains","ember_spirit_sleight_of_fist","batrider_firefly"];
console.log("单件(logit):", K.map(k=>nm(k)+" "+one(k).toFixed(3)).join(" | "));
for(let i=0;i<K.length;i++)for(let j=i+1;j<K.length;j++){const p=pair(K[i],K[j]); if(Math.abs(p)>0.05) console.log("  配合", nm(K[i]),"+",nm(K[j]), p.toFixed(3));}
const sets=[["batrider_sticky_napalm","npc_dota_hero_ember_spirit"],["ember_spirit_flame_guard","npc_dota_hero_ember_spirit"],["ember_spirit_flame_guard","batrider_sticky_napalm","npc_dota_hero_ember_spirit"],["ember_spirit_flame_guard"],["batrider_sticky_napalm"]];
for(const s of sets){const v=z(s)-z0; console.log("组合", s.map(nm).join("+"), "合计", v.toFixed(3), "约", pp(v), "个点");}
