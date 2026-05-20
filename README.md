# 4.3-log 仓库说明

这个仓库记录了两条并行工作线：

1. `v4.3` / `rt_v43_*` 在 HamGNN 硅材料任务上的修复与验收；
2. 以 `muon_ns` 对比 `adamw` 为唯一验收门槛的非 HamGNN benchmark 搜索。

当前最重要的判定规则只有一条：

- 只有当原始 `muon_ns` 明确优于 `adamw` 时，该 benchmark 才算通过。

## 当前总览

### 1. HamGNN 主线

- `test1-hamgnn-si/`
  - 修复后的 HamGNN `v4.3` 代码与硅任务完整结果
  - 被认可的主结果来自打包作业 `73239`
  - 最优方法是 `rt_v43_ns_b`
- `v4_3_hamgnn_repair_and_73239_audit.md`
  - 说明为什么只有 `73239` 这一轮结果可采信
- `v4_3_repair_73239_and_test2_pinn_audit.md`
  - 更完整的修复链说明，以及 `test2-pinn` 如何从占位内容修正为真实 PDE/PINN 路线

### 2. PDE / PINN 跟进线

- `test2-pinn/`
  - 单独的 PDE/PINN 跟进目录
  - 已包含真实 1D heat equation PINN benchmark
  - 比较方法为：
    - `adamw`
    - `muon_ns_adamw`
    - `rt_v43_stream_adamw`
    - `rt_v43_ns_adamw`

说明：

- `test2-pinn` 不是本轮“非 HamGNN benchmark 搜索”的主验收集合；
- 它的作用是给修复后的 `v4.3` 路线补一个真实 PDE/PINN 侧参考。

### 3. 非 HamGNN benchmark 搜索主线

这一部分是当前仓库里数量最多、也是本轮 GitHub 汇总的核心内容。

- `BENCHMARK_STATUS.md`
  - 最权威的总状态表
- `NON_HAMGNN_BENCHMARK_SUMMARY.md`
  - 非 HamGNN benchmark 的文字总结
- `ACCEPTED_BENCHMARKS_SNAPSHOT.md`
  - 当前已通过 benchmark 快照

这些目录主要是视觉/小规模文本/表格基准：

- `test10` 到 `test36` 中的大多数目录都属于这一组
- 其中真正被用于“`muon_ns` 是否比 `adamw` 更好”判定的，是各目录自己的 `STATUS.md` / `ANALYSIS.md` 与服务端落盘结果

## 各类目录怎么读

### 已通过的非 HamGNN benchmark

当前已经明确通过用户门槛的目录包括：

- `test11-vit-cifar10/`
- `test16-mlpmixer-cifar10-refine/`
- `test19-vit-cifar10-large/`
- `test21-vit-cifar10-p2-wide/`
- `test23-vit-cifar10-p2-deeper/`
- `test24-vit-cifar100-p2-wide/`
- `test26-vit-cifar100-p2-deeper/`
- `test29-stl10-vit/`
- `test30-tiny-imagenet-vit/`
- `test31-svhn-vit-p2-wide/`
- `test32-vit-cifar100-p2-longer/`
- `test33-stl10-vit-longer/`

这一组的共同特点是：

- 模型大多是 ViT 或 MLP-Mixer 一类矩阵参数占比较高的架构；
- 数据集都是真实视觉数据，不是玩具矩阵任务；
- `muon_ns` 的优势主要集中在 ViT 家族，尤其是更深、更宽、更长训练的设置。

### 已拒绝的非 HamGNN benchmark

当前明确失败的目录包括：

- `test10-pythia-tinystories/`
- `test12-mlpmixer-cifar10/`
- `test13-resmlp-cifar10/`
- `test14-gpt2-tinystories/`
- `test15-gpt2-tinystories-lrscan/`
- `test17-vit-cifar100/`
- `test18-gmlp-cifar10/`
- `test20-vit-cifar10-p2/`
- `test22-mlpmixer-cifar10-p2/`
- `test25-vit-cifar100-p2/`
- `test27-tabular-mlp-suite/`
- `test28-svhn-vit/`
- `test36-tiny-imagenet-mlpmixer/`

这组结果说明：

- `muon_ns` 并不是在所有架构上都稳定优于 `adamw`；
- 在文本、小型表格、部分 MLP-only 配方上，优势并不稳；
- 真正稳定给出正证据的，仍然主要是 ViT 家族。

### 仍在运行或暂不采信的目录

- `test35-tiny-imagenet-vit-deeper/`
  - 原始 `muon_ns` 对 `adamw` 已经明确占优
  - 后续大 sweep 原本继续进入 `rt_v43_stream`、`rt_v43_ns`、`rt_v6_fdt_metric`
  - 但该后续比较已按用户要求人工停止
  - 当前可采信结论是：这个 benchmark 已经足够支持 `muon_ns > adamw`
- `test34-flowers102-vit/`
  - 仍受数据集准备限制，不能当作已完成基准

## 目前可以下的结论

### 关于原始 `muon_ns`

从当前已经落地的非 HamGNN 结果看：

- 最强正证据集中在真实视觉任务；
- 最稳定的成功架构是 ViT；
- 当任务更接近“矩阵权重主导、视觉 token 化、较长优化过程”时，`muon_ns` 更容易显出优势。

### 关于 `v4.3`

从当前仓库证据看，`v4.3` 需要分成两层理解：

1. 在 HamGNN 主线上，修复后的 `v4.3` 已经有一轮被认可的完整结果，即 `test1-hamgnn-si/` 对应的 `73239`。
2. 在非 HamGNN 主线上，`test35` 已经完成了足够的 `muon_ns vs adamw` 验收部分；更后面的 `v4.3/v6` 比较被用户主动叫停，因此仓库当前只保留已经落地的部分结果，不再把它当作“待跑完”的阻塞项。

## 建议阅读顺序

如果想快速看清这批材料，建议按下面顺序：

1. 先看 `BENCHMARK_STATUS.md`
2. 再看 `NON_HAMGNN_BENCHMARK_SUMMARY.md`
3. 接着看：
   - `test31-svhn-vit-p2-wide/STATUS.md`
   - `test32-vit-cifar100-p2-longer/STATUS.md`
   - `test35-tiny-imagenet-vit-deeper/STATUS.md`
   - `test36-tiny-imagenet-mlpmixer/STATUS.md`
4. 如果关心 `v4.3` 修复链，再看：
   - `v4_3_hamgnn_repair_and_73239_audit.md`
   - `v4_3_repair_73239_and_test2_pinn_audit.md`

## 当前注意事项

这个 README 现在反映的是当前已经保留并准备上传的证据集：

- 原始 `muon_ns vs adamw` 验收结论已经收口
- `test35` 的后续 `v4.3/v6` sweep 按用户要求人工停止
- 因此本仓库的本轮汇总以“当前已落地并被保留的结果”为准
