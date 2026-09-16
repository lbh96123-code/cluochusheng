#!/usr/bin/env python3
"""从原始计数派生个人表现标签。口径分三类,按每个指标的物理含义选:

   ① 无量纲比值   KDA、击杀参与率、死亡占比 —— 本来就和时长无关,再除时间反而把
                  "这局节奏快"这种**对局属性**混进来,那不是个人能力
   ② 队内占比     伤害/治疗占本队的份额 —— 直接回答"这个人在队里承担了多少",
                  时长和对局强度自动抵消
   ③ 每分钟速率   gpm/xpm 本来就是速率;补刀用每分钟才有意义(补刀是累积量)

   八个标签:
     gpm            每分钟金钱                       ③
     xpm            每分钟经验                       ③
     lh_pm          每分钟补刀                       ③
     dmg_share      英雄伤害占本队份额 − 0.2          ②
     heal_share     治疗量占本队份额 − 0.2            ②
     kda            log1p((K+A)/max(D,1))            ①  取对数压掉零死亡的长尾
     kill_part      (K+A)/本队总击杀 − 0.2            ①  击杀参与率
     death_share    D/本队总死亡 − 0.2                ①
"""
import numpy as np

NAMES = ['gpm', 'xpm', 'lh_pm', 'dmg_share', 'heal_share', 'kda', 'kill_part', 'death_share']


def build(path='data_perf_raw.npz'):
    """返回 (labels (n,10,8) float32 已 z-score, NAMES)。"""
    d = np.load(path)
    raw, fields, dur = d['raw'], [str(x) for x in d['fields']], d['dur']
    g = {f: raw[:, :, i] for i, f in enumerate(fields)}
    n = len(raw)
    mins = (dur / 60.0)[:, None]

    def team_tot(x):        # 每人换成"本队 5 人之和",广播回 10 个座位
        return np.repeat(x.reshape(n, 2, 5).sum(2, keepdims=True), 5, axis=2).reshape(n, 10)

    K, D_, A = g['kills'], g['deaths'], g['assists']
    out = np.stack([
        g['gpm'],
        g['xpm'],
        g['lastHits'] / mins,
        g['heroDamage'] / (team_tot(g['heroDamage']) + 1e-6) - .2,
        g['heroHealing'] / (team_tot(g['heroHealing']) + 1e-6) - .2,
        np.log1p((K + A) / np.maximum(D_, 1.0)),
        (K + A) / np.maximum(team_tot(K), 1.0) - .2,
        D_ / np.maximum(team_tot(D_), 1.0) - .2,
    ], axis=2).astype(np.float32)

    flat = out.reshape(-1, out.shape[2])
    out = (out - flat.mean(0)) / (flat.std(0) + 1e-9)     # 统一 z-score,让 λ 对各指标一视同仁
    return out, NAMES


if __name__ == '__main__':
    L, nm = build()
    print(f'{L.shape[0]:,} 局 × 10 人 × {L.shape[2]} 个标签')
    flat = L.reshape(-1, L.shape[2])
    print(f'{"标签":13}{"均值":>8}{"标准差":>8}{"P1":>8}{"P99":>8}')
    for i, f in enumerate(nm):
        c = flat[:, i]
        print(f'{f:13}{c.mean():>8.3f}{c.std():>8.3f}{np.quantile(c,.01):>8.2f}{np.quantile(c,.99):>8.2f}')
