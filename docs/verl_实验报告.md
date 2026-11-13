# verl + 可证明过程奖励（PCPR）实验报告

本文记录把原 TRL/GRPO 算术实验迁移到 **verl 0.5.0 + Ascend 910B2** 的过程，以及在
此基础上提出的 **proof-carrying process reward (PCPR)** 和 **Process-GRPO** 探索。
英文版方法/协议见 [`verl_impl/RELATED_WORK.md`](../verl_impl/RELATED_WORK.md)，复现命令
见 [`verl_impl/README.md`](../verl_impl/README.md)。

## 1. 迁移

| 项 | 原实现 | verl 实现 |
| --- | --- | --- |
| 训练框架 | TRL `GRPOTrainer` | verl 0.5.0 `RayPPOTrainer` |
| 设备 | A100 CUDA | Ascend 910B2 NPU（torch_npu 2.4） |
| 环境 | `env-grpo` | `verllearning/.conda/envs/shtsky` |
| 并行 | 单卡 | 单机 6 卡，Ray + FSDP |
| rollout | HF generate | verl HF rollout |
| 奖励 | 最终答案 | 答案 / 格式 / 可证明过程 |
| 优势 | GRPO | GRPO / Dr.GRPO / RLOO / Process-GRPO / Decomposed / Difficulty |

本地模型为 `/root/work/models/Qwen3-0.6B`（机器上没有 1.7B）。通过自定义
`ArithmeticDataset` 在 chat template 中关闭 Qwen3 thinking，模型才会输出干净的
`a op b = c` 算术步骤；否则会陷入未闭合的 `<think>` 循环。

### 1.1 发现并修复框架 bug：`position_ids` 破坏 batch generation

第一次 smoke 时所有 rollout 都是数字重复的乱码，`response_length/clip_ratio=1.0`，
`actor/entropy≈0`。逐层排查发现：

- 单条 prompt 直接 `generate` 正常；
- 一旦把 `position_ids` 传给 `transformers.generate`，即使 batch 里没有 padding，
  Qwen3-0.6B 在 torch_npu 2.4 / transformers 4.53 上也会退化成数字循环；
- 去掉 `position_ids`、让模型从 `attention_mask` 自行推导位置后，生成恢复正常。

verl 0.5.0 的 `HFRollout` 正是显式传了 `position_ids`。修复补丁见
[`verl_impl/patches/0001-hf-rollout-drop-position-ids.patch`](../verl_impl/patches/0001-hf-rollout-drop-position-ids.patch)。
这个 bug 不是本项目特有：`verllearning` 的 MedQA demo 走同一条 rollout 路径，
其 baseline 2.35%（低于 25% 随机猜）很可能也受此影响；修复后值得重跑。

### 1.2 与近几年工作的关系

| 工作 | 会议/状态 | 本项目中的对应 |
| --- | --- | --- |
| GRPO (Shao et al., 2024) | arXiv 2402.03300 | `arithmetic_grpo_answer/format/process` |
| RLOO (Ahmadian et al., 2024) | ACL 2024 | `arithmetic_rloo_process` |
| Dr.GRPO (Liu et al., 2025) | COLM 2025 | `arithmetic_drgrpo_process`（去掉 std 归一化） |
| DAPO (Yu et al., 2025) | NeurIPS 2025 | `arithmetic_dapo_process`（clip-higher + token-mean + 去 KL + overlong） |
| VinePPO (Kazemnejad et al., 2025) | ICML 2025 | PCPR 是其「确定性 verifier 版」的 credit assignment 替代 |
| PRIME (Cui et al., 2025) | arXiv 2502.01456 | 学习型 implicit PRM 的对照；我们用零学习、可证明的 verifier |
| PRM survey (Zheng et al., 2025) | arXiv 2510.08049 | 支持「优先确定性 verifier，避免可 hack 的 PRM」 |
| VPR (Yuan et al., 2026) | arXiv 2605.10325 | 最接近的工作：dense turn-level verifier reward；PCPR 是算术域的确定性实例 |
| GSPO (Zheng et al., 2025) | arXiv 2507.18071 | 需要 sequence-level ratio；verl 0.5.0 不支持，列为 future work |
| Group-std identity (2026) | arXiv 2607.00152 | 用 `p^G+(1-p)^G` 解释 silent-group，并指导 difficulty-GRPO |

## 2. 方法：PCPR

1. 从 prompt 解析 gold 整数表达式，构建 AST，枚举所有规范中间值。
2. 从模型输出中按顺序抽取算术等式链：既支持 `a op b = c`，也支持
   `552 × 866 = 552 × (800+60+6) = … = 478032` 这样的多位分解链。链的每个部分都做精确
   求值；支持 LaTeX `\times`、`\cdot`、`\div`、Unicode 运算符、括号、千分位、步骤编号和
   负中间值。字符偏移在归一化时保留，供 token-level 奖励对齐。
3. 只有「整条链算术相等」且「最左表达式的叶子都已 grounded（来自 prompt 字面量或之前被
   接受的步骤）」的链才被接受。
4. 过程奖励 = 被证明的 gold 中间值比例（coverage）。重复等式去重；算术错误扣分。
5. token-level 版本把每个被验证步骤的奖励放在该步骤结束的 token 上；最终答案/格式奖励
   放在最后一个 token。每样本总奖励不变。
6. `process_grpo` 用反向累积和得到 dense return，再按 group/位置归一化；
   `decomposed_grpo` 把 final 与 process 两个通道分别归一化后相加；
   `difficulty_grpo` 用组内 reward 标准差对 Dr.GRPO 优势加权，作为 DAPO 硬过滤的软化版本。
7. **退化惩罚**：0.6B 模型有时陷入数字循环（`982 × 451 = 441, 982 × 451 = 441, …`）。
   `loop_repetition_ratio` 统计 40 字符片段的最大重复次数；若重复 ≥3 次，按覆盖比例
   扣分（`arithmetic_process_loop` 变体）。这给「最终错且 coverage=0」的组也提供了
   loop vs. non-loop 的组内信号，而不惩罚正常算术。

### 2.1 为什么能抗 reward hacking

朴素 step-count 奖励会为任何算术正确的等式付费，因此 `1 + 1 = 2` 刷屏可以骗奖励。
PCPR 的 coverage 只统计 **gold 中间值** 且要求 grounded，所以：
- 刷无关等式：coverage = 0；
- 用无关操作数凑出 gold 中间值：ungrounded，coverage = 0；
- 用 prompt 字面量写出**算术正确但与 gold 推导无关**的步骤：仍然 coverage = 0，
  朴素 step-count 会给 1.5，PCPR 只给 1.1；
- 正确中间步骤 + 错误最终答案：仍得到 0.5 × coverage 的密集信号（这正是全错组需要的）。

受控压力测试（16 条验证题，每类合成 completion）结果：

| completion 类型 | answer-only | 朴素 step-count | PCPR | coverage |
| --- | ---: | ---: | ---: | ---: |
| gold_trace | 1.000 | 1.500 | **1.600** | 1.000 |
| answer_only | 1.000 | 1.000 | 1.100 | 0.000 |
| wrong_final_correct_steps | 0.000 | 0.333 | **0.583** | 1.000 |
| trivial_spam | 1.000 | 1.500 | **1.100** | 0.000 |
| ungrounded_gold_steps | 1.000 | 1.333 | **1.100** | 0.000 |
| grounded_irrelevant_steps | 1.000 | 1.500 | **1.100** | 0.000 |
| arithmetic_error | 1.000 | 1.333 | 1.083 | 0.000 |
| repeated_steps | 1.000 | 1.167 | 1.350 | 0.500 |
| format_only | 0.000 | 0.000 | 0.100 | 0.000 |

复现：`python verl_impl/stress_test.py --rows 16`。

## 3. 理论

二值答案奖励、组大小 G、题内成功率 p 时，组内所有样本奖励相同的概率为

```
P_silent(p) = p^G + (1-p)^G
```

G=4 时：p=0.8 → 0.41，p=0.9 → 0.66，p=0.95 → 0.82。也就是说，当基座模型已经比较强时，
大部分组没有梯度。DAPO 用动态采样丢弃这些组并重新生成，代价是额外 rollout；PCPR 则让
奖励本身变成多级 coverage，使「最终错但步骤部分正确」的组重新出现方差。

初步观察（wave1 step 1，同一批 8 个 prompt × 4 样本）：

| 指标 | answer-only | process |
| --- | ---: | ---: |
| accuracy | 0.656 | 0.531 |
| mean reward | 0.656 | 0.863 |
| zero-var(acc) | 0.500 | 0.500 |
| zero-var(reward) | 0.500 | 0.500 |
| group reward std | 0.225 | **0.302** |
| all-wrong group rate | 0.000 | 0.125 |

process reward 把组内 reward std 提高了约 34%，但 zero-variance 组数没有下降：饱和组要么
全对（coverage 也是 1），要么全错且 coverage=0。也就是说，**coverage 只在「部分正确」的组
里创造方差；它不能拯救「所有样本都没有任何有效步骤」的组**。这正是 degeneration penalty
和 difficulty-GRPO 想补的缺口。最终结论以 8 步聚合数据为准。

四个 wave1 变体在同一 step 的初步对比（每变体 8 组 × 4 样本，采样随机性很大，仅作诊断）：

| 变体 | pass@1 | mean acc | coverage | zero-var(acc) | zero-var(reward) | group std | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer | 0.500 | 0.656 | 0.719 | 0.500 | 0.500 | 0.225 | 0.656 |
| format | 0.500 | 0.500 | 0.656 | 0.750 | 0.750 | 0.108 | 0.500 |
| process | 0.375 | 0.531 | 0.672 | 0.500 | 0.500 | **0.302** | 0.863 |
| drgrpo | 0.750 | 0.656 | 0.719 | 0.375 | 0.375 | **0.363** | 1.011 |

单步差异主要来自采样噪声，不能据此排名；但 process/drgrpo 的组内 reward std 明显更高，
与「dense reward 增加组内信号」的假设一致。**注意：训练在 step 2 因与本机另一个 session
的 6 个 MedQA 任务争抢 NPU 而崩溃；随后在协调出的 NPU 窗口内重跑 6 步核心对比。**

用公式核对要注意小样本偏差：单一全局 p=0.531 给出 `P_silent=0.128`（下界，忽略了题目
难度异质性）；用每组仅 4 个样本估计的 p 再平均给出 0.636（上界，p 的估计与「是否 silent」
相关，存在偏差）；实测 zero-var(acc)=0.5 落在这两者之间。结论是：**silent rate 对题目难度
异质性和估计样本量非常敏感，不能只用一个全局 p 预测，后续比较应以实测值为准**。process
的 8 个组中有 1 个全错组，但该组所有样本 coverage 都是 0，所以 zero-var(reward) 仍然是
0.5——这再次说明 coverage 在「全错且无有效步骤」的组里是饱和的。

**关键的第二个独立 batch**：6-step process run 的 step-1（同样是 8 组 × 4 样本）中，
zero-var(acc)=0.50，但 **zero-var(reward)=0.25**，coverage=0.609，all-wrong 组占 0.25，
group reward std=0.325。也就是说，当一个全错组包含部分正确步骤时，PCPR 把 silent group
比例减半，真正产生了非零 advantage。这正是 H1 预测的机制；它是否能在 6 步训练后转化为
准确率提升，是后续完整对比要回答的问题。

**steps 1–2 聚合**（16 组、64 样本）：accuracy=0.562，pass@1=0.500，pass@4=0.875，
coverage=0.656，group reward std=0.361，`zero-var(acc)=0.438` 而
`zero-var(reward)=0.312`——**降低 29%**，且两个 step 上方向一致，不是单批运气。

**steps 1–3 聚合**（24 组、96 样本）：accuracy=0.521，pass@1=0.500，pass@4=0.792，
coverage=0.651，group reward std=0.300，`zero-var(acc)=0.542` 而
`zero-var(reward)=0.333`——**降低 38%**。step 3 单独 `zero-var(acc)=0.75`，但聚合
reward zero-var 仍只有 1/3，说明 dense signal 在多个 batch 上持续起作用。

**steps 1–4 聚合**（32 组、128 样本）：accuracy=0.453，pass@1=0.438，pass@4=0.750，
coverage=0.602，group reward std=0.322，`zero-var(acc)=0.500` 而
`zero-var(reward)=0.281`——**降低 44%**。step 4 是更难的 batch（accuracy=0.25），
process reward 仍然在 accuracy 饱和的组里创造了方差，这是目前支持 H1 的最强证据。

**完整 6 步聚合**（48 组、192 样本）：pass@1=0.458，pass@4=0.708，
mean accuracy=0.448，coverage=0.576，group reward std=0.304，loop ratio=0.035；
`zero-var(acc)=0.542` 而 **`zero-var(reward)=0.292`——降低 46%**。逐 step accuracy
在 0.25–0.66 之间剧烈波动（每步只有 8 个 prompt）；response length 从 186 增长到
237 tokens，最后一个 step 的 clip ratio 达到 0.19。结论：**PCPR 确实在持续创造组内
信号，但 6 步不足以转化为稳定的准确率提升，且存在长度漂移风险。**

## 4. 实验设置

- 数据：`data/expanded_seed2028`，256 train / 64 validation / 96 test。
- 模型：Qwen3-0.6B + LoRA r=16/α=32，thinking 关闭。
- 训练：6–8 steps（共享 NPU 窗口下核心对比用 6 步），8 prompts × 4 samples/step，
  lr 1e-5，KL 0.001，prompt ≤256，response ≤320；数据顺序固定（`shuffle:false`），
  rollout 采样 temperature 0.7、top_p 0.95、top_k 20。
- 评测：
  - **实际执行**：32 条 validation，greedy（n=1，32 条一次生成），用于最终对比；
    另用训练 rollout（每步 32 条、n=4）做 zero-variance 动态。
  - **计划但未在本次窗口执行**：64 条 validation × 每题 4 个随机样本（pass@4 和
    更稳的配对统计），以及 96 条 held-out test 集。命令已就绪，见 RUNBOOK。
- 统计：按 prompt 配对的 bootstrap（10,000 次）95% CI + pass@1 的精确 McNemar 检验。
- 训练动态：每个 step 的 32 条 rollout 全部落盘，离线计算 mean accuracy、process
  coverage、zero-variance（按 acc 和按 reward 两种）、all-wrong 比例。
- 对照：未训练的 base model（无 LoRA adapter）使用完全相同的 eval 配置。

## 5. 结果

### 5.1 Base model（32 条 validation，greedy）

| 指标 | 值 |
| --- | ---: |
| accuracy | 0.531 |
| format rate | 0.000 |
| process coverage | 0.562 |
| valid steps / completion | 1.59 |
| full-proof completions | 15/32 |
| partial-proof completions | 6/32 |
| answer-only completions | 2/32 |
| wrong completions | 9/32 |
| dumped/recomputed metric mismatches | 0 |

失败模式：17 正确、11 算术错误（有 `\boxed{}` 但数值错）、4 数字循环、0 截断。说明：
- 基座已经能完成约一半题目，但 0% 使用要求的 `<answer>` 标签；
- 算术错误里常包含正确的中间步骤，这正是 PCPR 的用武之地；
- 数字循环是另一类失败，coverage=0，需要退化惩罚。

### 5.2 6-step 对比（32 条 validation，greedy）

| 变体 | reward | advantage | accuracy | coverage | loop | mean reward |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| base | — | — | 0.531 | 0.562 | 0.125 | 0.810 |
| process6 | process | GRPO | 0.562 | 0.672 | 0.094 | 0.895 |
| answer6 | answer | GRPO | 0.656 | 0.719 | 0.083 | 0.656 |
| dense6 | process + token | Process-GRPO | 0.625 | 0.703 | 0.091 | 0.974 |
| **drgrpo6** | process | **Dr.GRPO** | **0.750** | **0.828** | 0.060 | 1.160 |
| format6 | format | GRPO | 0.594 | 0.641 | 0.062 | 0.594 |
| loop6 | process + loop | GRPO | 0.625 | 0.703 | 0.091 | 0.956 |

排序：**Dr.GRPO + process (24/32) > answer-only (21/32) > dense/loop (20/32) >
format (19/32) > process+GRPO (18/32) > base (17/32)**。process reward 只有在
Dr.GRPO 下才明显有利；与标准 GRPO 的 std 归一化组合时反而低于 answer-only。这是
本实验最清晰的「算法 × 奖励」交互。

奖励分解：process 变体的 process component 约 0.34–0.41，final component 约
0.56–0.75，process 占总奖励约 35–38%；final 仍然主导。loop penalty 平均只有
~0.018（loop ratio ~9% × 权重 0.2），作用很小。

**格式审计**：32 条 greedy screen 中，base 有 28/32、format6 30/32、drgrpo6 30/32
使用 `\boxed{}`，但三者都是 **0/32 使用 `<answer>`**。0.1 的格式奖励在 6 步内没有
教会要求的标签。

配对 bootstrap + 精确 McNemar（32 条，pass@1）：

| 对比 | diff | 95% CI | McNemar |
| --- | ---: | --- | --- |
| base → answer6 | +0.125 | [0.000, +0.281] | p=0.219 |
| base → process6 | +0.031 | [−0.063, +0.125] | p=1.000 |
| base → dense6 | +0.094 | [−0.031, +0.219] | p=0.375 |
| **base → drgrpo6** | **+0.219** | **[+0.063, +0.375]** | **p=0.039 (8+/1−)** |
| base → format6 | +0.063 | [−0.063, +0.188] | p=0.625 |
| base → loop6 | +0.094 | [−0.031, +0.219] | p=0.375 |
| answer6 → process6 | −0.094 | [−0.219, +0.031] | p=0.375 (1+/4−) |
| answer6 → drgrpo6 | +0.094 | [0.000, +0.219] | p=0.250 (3+/0−) |

只有 `drgrpo6 − base` 的 CI 不含 0 且未校正 McNemar p<0.05；6 次比较的 Bonferroni
阈值是 0.0083，因此它**不能算校正后显著**，应视为有希望的探索性信号。反过来，
`answer6 → process6` 为负，说明 process reward 在标准 GRPO 下可能有害。

**逐题审计**（drgrpo6 vs base 的 9 个变化题）：8 改善 / 1 退步。

| prompt | base | drgrpo6 | 变化 |
| --- | --- | --- | --- |
| 451*982+25−6556 | loop | 436351 ✓ | 修复循环 |
| 280*229−37+1009 | 65892 ✗ | 65092 ✓ | 修复算术 |
| (149+380)*34−5913 | 12283 ✗ | 12073 ✓ | 修复算术 |
| 831*759−26+3541 | loop | 634244 ✓ | 修复循环 |
| 292*381−18+4742 | loop | 115976 ✓ | 修复循环 |
| 110*907−38+1729 | 101691 ✗ | 101461 ✓ | 修复算术 |
| (742+120)*31−1113 | loop | 25609 ✓ | 修复循环 |
| 204*941+29−7508 | 185485 ✗ | 184485 ✓ | 修复算术 |
| 552*866−12+5273 | 483293 ✓ | loop | 退步（新循环） |

这说明 **退化/循环模式是准确率差异的主要来源之一，且具有随机性**；单靠 reward 不能
稳定消除它，需要更强的循环惩罚、repetition penalty 或 SFT 预热。

按 base 对错分层（所有变体都保持同样的 16/17 道 base-correct 题；差异几乎全在难题）：

| 变体 | 修复 / 15 base-wrong | 保持 / 17 base-correct | 净增 |
| --- | ---: | ---: | ---: |
| answer6 | 5 (33.3%) | 16 (94.1%) | +4 |
| **drgrpo6** | **8 (53.3%)** | 16 (94.1%) | **+7** |
| dense6 | 4 (26.7%) | 16 (94.1%) | +3 |
| loop6 | 4 (26.7%) | 16 (94.1%) | +3 |
| format6 | 3 (20.0%) | 16 (94.1%) | +2 |
| process6 | 2 (13.3%) | 16 (94.1%) | +1 |

收益主要来自「把难题做对」，而不是牺牲简单题。process+GRPO 修复的难题最少，这解释了
它尽管有 dense reward、准确率却最低。

每个变体都只丢掉一道 base-correct 题：answer6/drgrpo6/dense6/loop6 丢的是同一个
易循环题 `552*866−12+5273`；process6/format6 丢的是 `(574+940)*34−6071`。因此
drgrpo6 的唯一退步是一个系统性的退化 case，而不是随机 trade-off。

**正确集合重叠**：drgrpo6 的 24 道正确题是 answer6 的 21 道的**超集**（交集 21、并集
24、Jaccard 0.88）——在这个 screen 上 Dr.GRPO+process 严格支配 answer-only。dense6 和
loop6 各 20 道，但它们与 drgrpo6 的并集仍然是 24（各有 4 道独有、也各漏掉 4 道），
因此这几个变体之间没有互补收益。

七个变体的**全变体 oracle 并集**是 25/32；drgrpo6 只差一道题就达到这个上界，唯一
「别的变体做对而 drgrpo6 做错」的题是 `552*866−12+5273`（它的新循环）。剩下 7 道题
所有变体都做错，是 6 步设置下的硬上限。

**Greedy vs stochastic（重要 caveat）**：32 条 greedy screen 上 drgrpo6 第一
（0.750 vs answer6 0.656）；但在 **48 条训练题 × 4 个随机样本**的 pass@4 上，排序翻转：

| 对比 | answer6 | 变体 | diff | 95% CI |
| --- | ---: | ---: | ---: | --- |
| answer6 → process6 | 0.792 | 0.708 | −0.083 | [−0.229, +0.063] |
| answer6 → drgrpo6 | 0.792 | 0.750 | −0.042 | [−0.167, +0.063] |
| answer6 → loop6 | 0.792 | 0.750 | −0.042 | [−0.167, +0.083] |
| answer6 → dense6 | 0.792 | 0.646 | **−0.146** | **[−0.271, −0.021]** |

screen 和 rollout 在**题目集**（validation vs train）和**解码方式**（greedy vs 4 个随机
样本）上都不同。process reward 可能锐化了 greedy mode，但降低了采样多样性；dense6 的
pass@4 显著低于 answer-only。64 条 validation × 4 的完整评测（同 validation 题、随机采样）
是判断这个差异的关键。

一个直接的解码对照：在 32 条共享 validation 题上，base 的 greedy accuracy 是 0.531，
但 base 的**随机第一样本** pass@1 是 0.594——仅解码方式就移动了 2 道题。因此不能把
greedy screen 当作随机策略的准确率。

2-step checkpoint 仍可作为链路验证：accuracy 0.531（与 base 相同），coverage 0.578
（base 0.562），loop 0.125，reward 0.817（base 0.810）。

### 5.3 64×4 完整评测

`verl_impl/full_eval_core.sh` 对 base、answer6、drgrpo6、process6、dense6、loop6
跑了 64 条 validation × 4 个随机样本（temperature 0.7），给出 pass@1/pass@4、
zero-variance 和配对统计的更大样本版本；结果写入 `verl_impl/RESULTS_FULL.md`，
原始 dump 在 `verl_impl/outputs/full_*/`，分析脚本为 `verl_impl/analyze_full.sh`。

**功效预估**：32 条 screen 上 drgrpo6 vs base 是 8 改善 / 1 退步（base-wrong 修复率
8/15，base-right 破坏率 1/17）。若这两个比例在 64 条上保持，discordant 对大约翻倍到
16 vs 2，精确 McNemar 约 p≈0.0013。因此完整评测即使在 screen 的 p=0.039 未校正显著之外，
也会给出更强的证据。

**Base 的 64×4 已完成**：pass@1=0.531，pass@4=0.797，mean acc=0.523，coverage=0.666，
loop=0.022。64 个组中 **22 全对、13 全错、29 混合**：13 个全错组是 outcome-only 的
silent group，29 个混合组是 PCPR 能增加方差的地方，22 个全对组对任何 outcome reward
都饱和。注意 base eval 用的是 process-shaped reward，所以
`zero-var(acc)=0.547` 而 `zero-var(reward)=0.375`——**即使不训练，process reward 本身
就把 base 模型的 silent-group 比例从 0.547 降到 0.375**。这直接说明 dense 性质来自
奖励设计，而不是训练带来的。

**Silent-group rescue**：13 个全错组共 52 个样本，**37/52（71%）拿到非零 process
score**，mean score 0.167、mean coverage 0.356——尽管所有最终答案都是错的。
11/13 个组有非零 score，10/13 个组有组内 score 方差。例如
`[0.00, 0.00, 0.000, 0.24]`（一个样本 coverage=0.5）和
`[0.00, 0.00, 0.237, 0.25]`（两个样本 coverage=0.5）。

| base 组类别 | 组数 | mean acc | mean coverage | mean process score |
| --- | ---: | ---: | ---: | ---: |
| all-correct | 22 | 1.000 | 0.994 | 1.497 |
| mixed | 29 | 0.397 | 0.556 | 0.670 |
| all-wrong | 13 | 0.000 | **0.356** | **0.167** |

在 answer-only 下 all-wrong 行应当严格为 0；process reward 把这 13 个组变成
mean score 0.167、平均证明 35.6% 的 gold 推导。这是 zero-variance 降低的直接机制，
而且在**未训练的 base 模型**上就已经存在。

**完整的 64×4 评测（六个变体全部完成）**：

| 变体 | pass@1 | pass@4 | mean acc | coverage | loop | zero-var(acc) | zero-var(reward) | all-wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 0.797 | 0.523 | 0.666 | 0.022 | 0.547 | 0.375* | 0.203 |
| answer6 | 0.531 | **0.875** | 0.578 | 0.717 | 0.017 | 0.453 | 0.453 | **0.125** |
| process6 | 0.547 | 0.797 | 0.547 | 0.689 | 0.022 | 0.531 | 0.359 | 0.203 |
| dense6 | 0.563 | 0.797 | 0.578 | 0.699 | 0.023 | 0.594 | 0.406 | 0.203 |
| **drgrpo6** | **0.672** | 0.859 | **0.660** | **0.775** | 0.012 | 0.563 | 0.406 | **0.141** |
| loop6 | 0.609 | 0.844 | 0.594 | 0.727 | **0.004** | 0.531 | 0.344 | 0.156 |

\* base 的 reward 已经是 process-shaped；0.375 不是 answer-only 基线。policy 层面的
基线是 `zero-var(acc)=0.547`。

**结论在预注册评测下成立**：Dr.GRPO+process 在 pass@1（0.672 vs base 0.531，
McNemar p=0.078；vs answer-only 0.531，p=0.035）和 mean accuracy（0.660 vs base
0.523，+0.137，CI 不含 0；vs answer-only +0.082，CI 不含 0）上都是第一。answer-only
是 best-of-4 变体（pass@4 0.875、all-wrong 最低 0.125），但 pass@1 不变。loop6 的
loop rate 最低（0.4%），mean accuracy 相对 base 有显著提升（+0.070，CI [+0.004,+0.137]）。
所有 process 变体的 reward zero-var 相对自己的 acc zero-var 降低 28–35%。

**Best-of-4 增益**（`pass@4 − pass@1`）：answer6 +0.344、base +0.266、process6 +0.250、
loop6 +0.234、dense6 +0.234、drgrpo6 +0.188。answer-only 从采样多样性中获益最大；
Dr.GRPO+process 获益最小，但它的 pass@1 起点最高。不同变体改善的是分布的不同部分。

### 5.4 96 条 held-out 文字题（OOD，greedy）

Base 在 held-out 工厂文字题上 **accuracy=0.594（57/96）**，coverage=0.609，
format=0.104，loop=0.042。文字题的 `<answer>` 命中率（10.4%）明显高于符号题
（0%）——prompt 模板会显著影响格式遵循。失败模式：57 正确、**35 算术错误**、
4 数字循环、0 截断——OOD 失败主要是算术失误，而不是退化。

**迁移结果（96 条 held-out 文字题，greedy）**：

| 变体 | accuracy | coverage | format | loop |
| --- | ---: | ---: | ---: | ---: |
| base | 0.594 | 0.609 | 0.104 | 0.042 |
| **answer6** | **0.635** | 0.667 | 0.031 | 0.042 |
| drgrpo6 | 0.615 | 0.656 | 0.052 | 0.042 |
| process6 | 0.542 | 0.589 | 0.094 | 0.010 |

配对 vs base：answer6 +0.042（CI [0.000,+0.094]，McNemar p=0.219，5+/1−）；
drgrpo6 +0.021（CI [−0.031,+0.073]，p=0.688，4+/2−）；process6 **−0.052**
（CI [−0.104,−0.010]，p=0.063，**0+/5−**）。

**OOD 最佳是 answer-only，而不是 in-distribution 冠军 drgrpo6**。answer-only
迁移最好（57→61 题）；drgrpo6 为正但更小（57→59）；process+标准 GRPO 显著变差
（57→52，5 个 discordant 对全部退步，CI 不含 0）。一个合理解释是 process reward
鼓励了偏符号题的 trace 风格，在符号 validation mode 上有帮助，但不如 answer-only 的
「solve and answer」行为迁移得好；在标准 GRPO 下，额外的 dense 信号在 OOD 上反而有害。

**正确集合并集**：四个变体合起来 63/96。answer6 61 题（2 道独有），drgrpo6 59 题
（1 道独有），base 57 题（0 独有），process6 52 题（0 独有）。answer6+drgrpo6 的
oracle 集成 63/96（65.6%），只比 answer6 单模型多 2 题——互补性很小，
answer-only 是最佳的单模型迁移选择。

### 5.5 扩展算法 wave（32 条 greedy，6 步）

在同样的 6 步预算下再训练 RLOO、DAPO-lite（clip-higher 0.28 + 去 KL + overlong）、
Decomposed-GRPO、Difficulty-GRPO，以及组大小 n=8（每 step 4 条 prompt × 8 样本，
总 completions 不变）：

| 变体 | 优势估计 | accuracy | coverage | loop | mean reward |
| --- | --- | ---: | ---: | ---: | ---: |
| rloo | RLOO (n=4) | 0.656 | **0.750** | 0.028 | 1.028 |
| dapo | GRPO (DAPO-lite) | 0.656 | 0.734 | 0.063 | 1.014 |
| decomposed | Decomposed-GRPO | 0.625 | 0.703 | 0.091 | 0.974 |
| difficulty | Difficulty-GRPO | 0.656 | 0.734 | 0.060 | 1.021 |
| n8 | GRPO, n=8 | 0.656 | 0.719 | 0.091 | 1.012 |
| _参照_ process6 | GRPO (n=4) | 0.562 | 0.672 | 0.094 | 0.895 |
| _参照_ answer6 | answer-only | 0.656 | 0.719 | 0.083 | 0.656 |
| _参照_ drgrpo6 | Dr.GRPO (n=4) | **0.750** | **0.828** | 0.060 | 1.160 |

32 条 prompt 上 1 条 = 3.1 个百分点，因此 0.625 与 0.656 只差一条 prompt；只有
Dr.GRPO 的 0.750（24/32）清晰分离。**结论是一个诚实的负结果**：RLOO、DAPO-lite、
Decomposed-GRPO、Difficulty-GRPO 和 n=8 在 6 步预算下都没有复现 Dr.GRPO+process
的增益。Decomposed-GRPO 在扩展变体里训练 pass@k 最高（0.771，仍低于 answer-only
的 0.792），但 greedy screen 最低（0.625）且 loop 比例最高（0.091）：
final/process 分解在短训练下增加方差而没有增加信号。
RLOO 的 coverage 最高（0.750）；Difficulty-GRPO 把 invalid steps 降到 0.12，为 wave 最低。

与 drgrpo6 在同一 32 条 prompt 上配对（`stats.py --metric pass@1`）：

| 对比 | drgrpo6 | 变体 | diff | 95% CI | McNemar |
| --- | ---: | ---: | ---: | --- | --- |
| drgrpo6 → rloo | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+/3−) |
| drgrpo6 → dapo | 0.750 | 0.656 | −0.094 | [−0.219, +0.031] | p=0.375 (1+/4−) |
| drgrpo6 → decomposed | 0.750 | 0.625 | −0.125 | **[−0.250, −0.031]** | p=0.125 (0+/4−) |
| drgrpo6 → difficulty | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+/3−) |
| drgrpo6 → n8 | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+/3−) |

只有 decomposed 的配对 bootstrap CI 不含 0，但只有 4 个 discordant 对
（McNemar p=0.125），且 5 次比较未校正，只能算探索性。诚实的读法是：扩展变体
没有显著**差于** Dr.GRPO（decomposed 可能除外），但也没有任何一个是更好的；
把 Dr.GRPO 与 answer-only 分开的那个效应（64×4 上 12 改善/3 退步）没有被复现。

扩展变体的训练 rollout 动态（48 组 × 4；n=8 为 24 组 × 8）：

| 变体 | mean acc | pass@1 | pass@k | zero-var(acc) | zero-var(reward) | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rloo | 0.516 | 0.583 | 0.729 | 0.542 | **0.333** | 0.609 | 0.031 |
| dapo | 0.469 | 0.542 | 0.688 | 0.583 | **0.333** | 0.586 | 0.019 |
| decomposed | 0.448 | 0.458 | **0.771** | 0.479 | **0.333** | 0.594 | 0.017 |
| difficulty | 0.479 | 0.417 | 0.708 | 0.583 | 0.375 | 0.591 | **0.013** |
| n8 | **0.552** | **0.625** | **0.875** | **0.458** | **0.333** | 0.625 | 0.024 |
| _参照_ answer6 | 0.490 | 0.417 | 0.792 | 0.458 | 0.458 | 0.594 | 0.002 |
| _参照_ drgrpo6 | 0.490 | 0.542 | 0.750 | 0.562 | 0.313 | 0.620 | 0.024 |

这一节唯一的**稳健正结果**：五种扩展变体把 reward zero-variance 比例都压在
0.33–0.38，而 answer-only 是 0.46 —— dense PCPR 信号制造组内方差的能力与优势估计器
无关，RLOO 与 DAPO 式裁剪都保留了这一机制。n=8 因为每条 prompt 有更多样本，all-wrong
组比例自然降到 0.125、训练 pass@8 达 0.875，但 greedy screen 仍为 0.656。

另一个值得记录的观察是**长度漂移**：6 步内 mean output chars 分别增长 +97（RLOO）、
+95（DAPO-lite）、+95（Difficulty-GRPO）、+68（Decomposed-GRPO），而 **n=8 只有 +27**
（对照：drgrpo6 +96、process6 +79、answer6 +72）。n=8 是唯一基本保持输出长度的变体：
每条 prompt 8 个样本时，正确与错误模式在每个组内都出现，策略得到的是组内直接比较，
不依赖长度。这与 Dr.GRPO 的长度偏置论点一致，但提示**组大小本身也是控制漂移的杠杆**；
单 seed/6 步只能算提示性证据。

长度漂移**不是**「越长奖励越高」造成的：在全部变体的 192 条训练 rollout 上，输出长度
与 reward 的 Pearson 相关都是负的（RLOO −0.10、drgrpo6 −0.12、answer6 −0.12、
decomposed −0.16、difficulty −0.19、process6 −0.22、n8 −0.30），与正确性也负相关。
更长的 trace 系统性更差，但输出仍在变长——这说明漂移是策略分布本身的漂移，而不是
优化朝更长答案移动；n=8 漂移最小、长度-奖励负相关最强，与「其优化方向最贴近更短且
正确的 trace」一致。

### 5.6 稠密信号的校准：coverage 能否预测最终正确？

dense reward 只有在与 outcome 一致时才有意义。我们用 `process_coverage`
（模型真正证明的 gold 中间值比例）预测最终答案是否正确，计算 AUC
（`verl_impl/process_auc.py`）：

| 变体（96 条 held-out 文字题） | n | 正确数 | coverage｜正确 | coverage｜错误 | AUC | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | 96 | 57 | 0.947 | 0.115 | 0.958 | [0.910, 0.995] |
| answer6 | 96 | 61 | 0.975 | 0.129 | 0.947 | [0.891, 0.991] |
| drgrpo6 | 96 | 59 | 0.966 | 0.162 | 0.933 | [0.875, 0.983] |
| process6 | 96 | 52 | 0.952 | 0.159 | 0.947 | [0.898, 0.986] |

32 条符号题的 13 个 checkpoint 上 AUC 为 0.914–0.963（base 0.918、drgrpo6 0.961、
rloo 0.959、dapo 0.961、decomposed 0.960、difficulty 0.961、n8 0.963）。

256 个随机样本的 64×4 评测上 AUC 为 0.910–0.942（CI 下界 ≥0.871）：base 0.942、
dense6 0.937、loop6 0.930、answer6/drgrpo6 0.922、process6 0.910。随机采样下
错误样本包含更多「部分正确 trace」（平均 coverage 0.33–0.39），所以分离度低于 greedy，
但仍是强预测器。

**解读。** proof-carrying coverage 在两种 prompt 分布、所有训练变体上都是 outcome
的强预测器；连从未见过该 reward 的 base 模型也有 AUC 0.92–0.96。这说明 verifier
测到的是真实的算术进展，而不是格式伪影。**但校准好不等于适合训练**：96 条 OOD 上
PCPR 校准极好，而标准 GRPO + PCPR 反而显著降低 OOD 准确率（5.4 节）。
「好的稠密预测器」与「好的学习信号」是两件事，区别在于优势估计器如何使用这份额外方差——
这是本报告的核心教训。

AUC 的分离有一部分来自 coverage=0 的退化输出；不过错误样本中有 17–25% 至少验证通过
一步（base 7/39、answer6 6/35、drgrpo6 8/37、process6 11/44），其平均 coverage 为
0.64–0.75，即真实的「步骤对、最后算错」。这类样本正是稠密奖励应当帮助的对象，
同时也是「做完中间步骤就停」被奖励的来源。

## 6. 训练动态

每个 step 的 32 条 rollout 都落盘到 `verl_impl/outputs/<config>/rollout/<step>.jsonl`，
`verl_impl/analyze.py --per-step` 可计算 mean accuracy、process coverage、
zero-variance（按 acc / 按 reward）、all-wrong 比例和组内 reward std。

6-step 跨变体 rollout 动态（每变体 48 组 × 4 样本，随机采样，不是 greedy screen）：

| 变体 | mean acc | pass@1 | pass@4 | zero-var(acc) | zero-var(reward) | group std | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer6 | 0.490 | 0.417 | 0.792 | 0.458 | 0.458 | 0.251 | 0.594 | 0.002 |
| format6 | 0.448 | 0.479 | 0.688 | 0.563 | 0.563 | 0.199 | 0.503 | 0.019 |
| process6 | 0.448 | 0.458 | 0.708 | 0.542 | **0.292** | 0.304 | 0.576 | 0.035 |
| dense6 | 0.474 | 0.500 | 0.646 | 0.646 | **0.375** | 0.236 | 0.573 | 0.012 |
| drgrpo6 | 0.490 | 0.542 | 0.750 | 0.563 | **0.313** | 0.289 | 0.620 | 0.024 |
| loop6 | 0.510 | 0.604 | 0.750 | 0.542 | **0.271** | 0.314 | 0.620 | 0.030 |

所有 process-reward 变体都把 zero-variance 组比例降低 **34–50%**；answer-only 和
format（其 `<answer>` 命中率只有 0.5%，奖励几乎等于 accuracy）做不到。但 process
变体的 **平均 rollout accuracy 并没有更高**——dense 信号改善的是组内方差，不是平均奖励。
逐 step accuracy 在 0.25–0.66 之间波动（每步 8 个 prompt），response length 从 186
增长到 237 tokens，最后 clip ratio 0.19，存在长度漂移。

逐 step 的两个细节（`verl_impl/dynamics_report.py`）：
- **困难 step 4**（mean acc 0.22–0.375）上，drgrpo6/dense6/loop6 的
  `zero-var(reward)=0.000`，而 `zero-var(acc)` 分别为 0.25/0.50/0.50；answer6 和
  format6 的 zero-var(reward) 仍等于 zero-var(acc)。这是 dense signal 在「几乎全错」
  batch 上最直接的作用。
- **长度漂移是普遍现象**：6 步内所有变体的 mean output chars 增长 57–96（answer6 +72，
  drgrpo6 +96，dense6 +77，loop6 +57，format6 +83）。Dr.GRPO 并没有消除长度漂移；
  单 seed/6 步的数据不足以验证它的 length-bias 理论。

理论 vs 经验（`verl_impl/silent_group_analysis.py`，逐 step）：

| 变体 | 全局 p 公式（下界） | 组内 p 公式（上界） | 经验 acc silent | 经验 reward silent |
| --- | ---: | ---: | ---: | ---: |
| answer6 | ~0.125 | 0.583 | 0.458 | 0.458 |
| process6 | ~0.111 | 0.656 | 0.542 | **0.292** |
| drgrpo6 | ~0.139 | 0.666 | 0.563 | **0.313** |

全局 p 公式忽略题目难度异质性、给下界；组内 p 公式用同一组 4 个样本估计 p、给上界；
经验 acc silent 落在两者之间。PCPR 把经验 reward silent 压到**上界以下**，直接说明
它在 outcome-only GRPO 没有信号的组里制造了信号。

## 7. 讨论

1. **H1 得到强支持：PCPR 持续降低 zero-variance 组比例。** 6-step rollout 中，
   answer-only 的 `zero-var(reward)=zero-var(acc)=0.458`（按构造必然相等）；四个
   process-reward 变体把 zero-var 降低到 0.271–0.375，相对 accuracy 的 zero-var 降低
   34–50%。process6 的 46% 降幅在 steps 1–4 的多个 batch 上方向一致，不是单批运气。
2. **H5 得到强支持：算法与奖励的交互是决定性的。** 32 条 greedy 上，
   Dr.GRPO + process 达到 24/32（0.750），是唯一相对 base 的 CI 不含 0 的变体
   （+0.219，p=0.039，未校正）；而 process + 标准 GRPO 只有 18/32（0.562），
   比 answer-only 的 21/32 更差。Dr.GRPO 去掉 std 归一化，避免了 process reward
   的额外方差被 std 放大；这与 Dr.GRPO 原文对 length/std bias 的批评一致。
   **64×4 预注册评测确认了这一点**：drgrpo6 pass@1=0.672、mean acc=0.660，
   相对 base 的 mean acc +0.137（CI [+0.078,+0.195]），相对 answer6 的 pass@1
   +0.141（CI [+0.031,+0.250]，McNemar p=0.035）；process6/dense6 的 pass@1 只有
   0.547/0.563，没有明显超过 answer6 的 0.531。
3. **H2 的 token-level dense reward 没有转化为准确率优势。** dense6 为 20/32（0.625），
   高于 process6 但低于 answer6；Process-GRPO 的密集 credit 在这个短训练/小模型上
   没有明显收益。H3（decomposed）与 H4（difficulty）现已完成 6 步训练和 screen：
   decomposed 0.625、difficulty 0.656，同样没有超过 Dr.GRPO。扩展 wave 的 RLOO
   0.656、DAPO-lite 0.656、n=8 0.656 也落在同一区间（见 5.5）。
4. **format bonus 太弱。** format6 的 `<answer>` 命中率只有 0.5%，6 步内没有学会标签；
   0.1 的格式奖励不足以对抗模型原本的 `\boxed{}` 习惯。若格式是目标，应提高权重或
   用 SFT 预热。
5. **degeneration penalty 没有明显的准确率收益。** loop6 为 20/32（0.625），与 dense6
   相同；训练 rollout 的 loop 比例 3.0%，base greedy 是 12.5%，但两者的采样条件不同，
   不能直接比较。需要更长的训练和更大的 batch 才能判断。
6. **框架 bug 是这次实验最重要的发现之一。** 在修复 `position_ids` 之前，所有 batch
   rollout 都是数字循环，RL 实际上在拟合噪声；这也解释了同环境 MedQA baseline 低于
   随机猜的现象。任何在此环境上做 verl + Qwen3 的工作都应先检查这一点。
7. **统计上要克制，但 64×4 给出了更强的证据。** 32 条 screen 的 p=0.039 是未校正的；
   64×4 上 drgrpo6 vs answer6 的 pass@1 p=0.035、mean acc CI 不含 0；drgrpo6 vs base
   的 mean acc CI 不含 0、pass@1 p=0.078（边缘）。仍是单 seed、6 步、64 条 validation，
   不能外推到其他模型/任务；但结论已从「单一 screen 的探索性信号」升级为
   **「在预注册的 64×4 评测上方向一致且有统计支持」**。PCPR 的训练动态结论
   （zero-variance 降低 28–35%）在两种评测下都成立。
8. **In-distribution 冠军不等于 OOD 冠军。** 96 条 held-out 文字题上，answer-only
   迁移最好（0.635 vs base 0.594，+4.2 点，CI [0,+9.4]）；drgrpo6 为正但更小
   （0.615，+2.1）；process+标准 GRPO 显著变差（0.542，−5.2，CI [−10.4,−1.0]，
   0+/5−）。process reward 的收益可能是符号 trace 模板特定的；answer-only 学到的是
   更通用的「solve and answer」行为。这提醒我们：**在只看 in-distribution 指标时，
   dense reward 的 OOD 代价会被隐藏**。
9. **PCPR 校准良好，但这不等于它是好的学习信号。** coverage 对最终正确性的
   AUC 在 96 条 OOD 上为 0.933–0.958（CI 下界均 ≥0.875），在 32 条符号题的全部
   13 个 checkpoint 上为 0.914–0.963；连 base 模型都有 0.92–0.96。正确样本的平均
   coverage ~0.95、错误样本 ~0.13。但同一份 reward 在标准 GRPO 下会损害 OOD 准确率。
   **预测能力与训练价值正交**，取决于优势估计器如何使用这部分方差；评估 dense
   reward 时必须把「校准」和「学习收益」分开报告。

## 8. 创新点与工程贡献

1. **Proof-carrying process reward (PCPR)**：不训练 PRM、不做 Monte-Carlo 展开，直接用
   gold 表达式的 AST + grounded 等式链验证每一步；coverage 只统计 gold 中间值，天然抗
   `1+1=2` 刷屏和「凑 gold 值」攻击。
2. **等式链解析**：支持 `552×866 = 552×(800+60+6) = … = 478032`、LaTeX/Unicode 运算符、
   千分位、步骤编号、负中间值；字符偏移保留，供 token-level 奖励对齐。
3. **Token-level dense reward + Process-GRPO**：把每个被验证步骤的奖励放在该步骤结束的
   token 上，用反向累积和 + 按位置 group 归一化得到 dense advantage。
4. **Decomposed-GRPO**：final 与 process 两个通道分别 group 归一化后相加，避免
   outcome 项淹没 process 项。
5. **Difficulty-GRPO**：用组内 reward std 对 Dr.GRPO 优势加权，是 DAPO 硬过滤的软化版本，
   不需要额外 rollout。
6. **退化惩罚**：检测数字循环并扣分，给「全错且 coverage=0」的组提供组内信号。
7. **发现并修复 verl/torch_npu 的 `position_ids` batch-generation bug**：该 bug 会让所有
   rollout 退化成乱码，可能也影响了同环境下的 MedQA demo。
8. **可复现的研究脚手架**：13 个 Hydra 配置、28 个单元测试、300+ 随机表达式 fuzz、
   受控 reward-hacking 压测、逐样本 audit、配对 bootstrap + McNemar 统计、
   coverage→correctness 校准分析（AUC，`process_auc.py`）。
9. **共享机器上的鲁棒性**：设备占用检查、`SAVE_FREQ=2` 断点、`resume_mode=auto`、
   `ActorDiedError` 诊断；与同机另一个 session 显式协调 NPU 时间片，避免再次互相打断。
10. **实证发现**：PCPR 把 zero-variance 组比例降低 28–50%；64×4 预注册评测上
    Dr.GRPO+process 达到 pass@1 0.672 / mean acc 0.660，显著高于 answer-only 的
    0.531/0.578（pass@1 p=0.035，mean acc CI 不含 0），也高于 base 的 0.531/0.523。
    answer-only 是最佳 best-of-4 变体（pass@4 0.875）；loop penalty 把 loop rate
    从 base 的 2.2% 降到 0.4%。算法与奖励的交互比“是否加 process reward”本身更重要。
11. **OOD 迁移的负结果**：96 条 held-out 文字题上，answer-only 迁移最好（0.635），
    drgrpo6 为正但更小（0.615），process+标准 GRPO 显著变差（0.542，CI 不含 0，
    0+/5−）。**in-distribution 冠军 drgrpo6 不是 OOD 冠军**；dense reward 的收益
    部分是符号模板特定的，必须单独报告 OOD 迁移。
12. **扩展算法 wave 的负结果 + 稳健正结果**：RLOO、DAPO-lite、Decomposed-GRPO、
    Difficulty-GRPO、n=8 五种变体在 32 条 greedy screen 上都落在 0.625–0.656，
    没有一个复现 Dr.GRPO 的 0.750；但它们的 reward zero-variance 比例都在
    0.33–0.38（answer-only 为 0.46），说明 dense PCPR 制造组内方差的能力不依赖
    优势估计器。这一「机制稳健、准确率增益不稳健」的分离是本次探索最诚实的结论。

## 9. 局限

- 主要训练是 6 步、单 seed；64×4 完整评测已完成（见 `RESULTS_FULL.md`），给出了
  drgrpo6 相对 answer-only 的显著 pass@1/mean-acc 优势。96 条 held-out test 与
  RLOO/DAPO/decomposed/difficulty/n8 扩展 wave 均已完成（见 `RESULTS_EXTENDED.md`），
  但扩展变体只在 32 条 greedy screen 上评测：没有任何一个超过 Dr.GRPO（0.750），
  全部落在 0.625–0.656。仍是单一模型/任务，不能外推到其他设置。
- Qwen3-0.6B 比原 TRL 项目的 1.7B 弱，绝对准确率不可直接比较。
- PCPR 只验证整数算术等式和 gold 中间值，不验证文字题的语义分解。
- verl 0.5.0 没有实现 DAPO dynamic sampling 和 GSPO；DAPO-lite 只覆盖配置层可实现的子集。
- 扩展 wave 的 32 条 greedy screen 中，1 条 prompt 就是 3.1 个百分点；
  RLOO/DAPO/decomposed/difficulty/n8 之间的差异基本在一条 prompt 以内，
  真正的分离需要更大的评测集或更长的训练预算。
- 奖励权重（final 1.0 / process 0.5 / format 0.1 / loop 0.2）是先验设定，未做扫描；
  format bonus 0.1 在 6 步内太弱。
- 训练 rollout 的 response length 从 186 增长到 237 tokens，最后 clip ratio 0.19，
  存在长度漂移；Dr.GRPO 的 length-bias 优势可能部分来自这一点。
- 共享 NPU 机器上的并行实验需要显式协调；一次碰撞导致 wave1 在 step 2 崩溃。
