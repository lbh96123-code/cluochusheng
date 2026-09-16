# AD 选技模型 · 起步包说明（不需要任何账号）

> 本仓库 = 起步包 starter_kit_20260916 的内容（数据 + 脚本 + 基线权重 + 文档）。全量镜像见下方链接。


目录镜像: http://43.130.62.185/adres-3fd0da4583c67d3a/  （浏览器可直接点开、下载任意文件）

## 一次拉下整个目录（7.1 GB）
    wget -r -np -nH --cut-dirs=1 -R "index.html*" http://43.130.62.185/adres-3fd0da4583c67d3a/

## 只拿起步包（约 280 MB, 够复现主模型训练和评测）
    curl -O http://43.130.62.185/adres-3fd0da4583c67d3a/starter_kit_20260916.tgz
    tar xzf starter_kit_20260916.tgz && cd starter_kit

包内: 训练数据 data_pure_d.npz(编号+胜负) / data_perf_raw_d.npz(赛后8项, 仅训练期辅助) / 两半切分 *_hA *_hB /
      件名表 pairaudit/model_pair.npz / 基线模型与两半模型 model_comp_K96_lam10.0_b8_sgn_win_*.npz /
      脚本 bench.py npcompat.py perflabels.py exp_comp.py exp_comp_pf2.py exp_perf_abs.py seatcorr.py scorecard.py /
      文档 交接文档.html README.md EXPERIMENT.md doc_share/(三份方法论 PDF)

## 环境
    python3 -m pip install numpy scipy      # 只需要这两个, 无 GPU, 2 GB 内存够用(脚本已分块)

## 复现基线（约 30 分钟, 2 核）
    BENCH_PURE=data_pure_d.npz PERF_RAW=data_perf_raw_d.npz PERF_Z=data_perf_d.npz \
      SIGNED=1 WITHIN=1 EXT_TAG=_d PATIENCE=6 python3 exp_comp.py 96 40 8e-4 10.0 8
    # 期望: RESULT 行 logloss ≈ 0.5777, AUC ≈ 0.762 (测试段 69,914 局)
    # 注: PERF_Z 只在 PERFMODE=share 时读取, 没下 data_perf_d.npz 也能跑默认口径

## 座位分 vs 个人表现
    python3 seatcorr.py comp_K96_lam10.0_b8_sgn_win_d

## 从头看
    交接文档.html(总览) → EXPERIMENT.md(实验契约与已判负清单) → doc_share/半正定这道锁.pdf(模型核心)
