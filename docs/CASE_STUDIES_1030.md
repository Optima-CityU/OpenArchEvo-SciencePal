# 案例素材（10/30，供视频 / 幻灯片 / 说明直接引用；全部来自真实运行记录）

## 案例 1：一行之差（冠军 vs 旧输出门）

规范化后的核心前向差异（两个程序其余前向计算相同；另一处差异是 MLP 脉冲统计的返回方式，训练脚本只用它记录发放率，不进入损失）：

```diff
        o, spikes_o = self.o_proj(o)
        gate, spikes_gate = self.gate_proj(hidden_states)   # gate = W · spike(h)
-       gate = torch.sigmoid(gate)                            # 旧输出门 r03a-263bee：30 轮 73.54
        o = o * gate                                          # 冠军 r02a-e29092：30 轮 68.95
```

- 2 轮（搜索打分）：117.66 vs 117.70；30 轮：73.54 vs 68.95（各 3 种子）；测试 PPL 65.55 vs 61.65。图 `figs/F2_one_line.png`。
- 门值是由脉冲输入经线性映射得到的连续值（"脉冲驱动的乘性门"），不是二值脉冲。

## 案例 2：一张经验卡的一生

1. GPU 证据：E12 / E18 / E20 / E21（投影后 sigmoid 门在 30 轮落后；去掉 sigmoid 的门 69.02）。
2. 写成卡片（Skill v0.5.1，节选）：

## Experience cards (v0.5.1): GPU-validated lessons from this week's experiments

These cards come from controlled GPU experiments (WikiText-2, SLTT neurons, seq 128, the `sp-wt2-sltt-s128-*` protocols).
They are evidence about **small spiking DeltaNet-family blocks trained for 30 epochs**; treat them as strong priors, not laws,
and say in `notes` when you deliberately go against one. Cite the card id in `notes` when it shaped your design.

| id | pattern | situation | evidence (30-epoch val PPL, lower is better) | scope |
|---|---|---|---|---|
| C1 | **avoid** | adding a *sigmoid* (attenuate-only) output gate **after** `o_proj` | looks best at 2 epochs (117.66) but is worse than DeltaNet at 30 epochs: 73.54 (3 seeds) vs 71.82 (6 runs); train/val gap grows from epoch ~8 (overfits) | E12, E18, E20 |
| C2 | **validated** | if you gate the output, gate **before** `o_proj` (per head, inside the value path) | moving the same gate before `o_proj`: 70.72 (2 seeds); DeltaNet's own `use_gate` (gated norm): 70.66 (3 seeds); another pre-projection gate + state gate: 70.36 (1 seed) | E21 i4, E18, E20 |
| C3 | **validated** | gates that can amplify or are identity-centred (e.g. `1 + tanh`, near-identity init, no sigmoid) | identity-centred: 71.63; `1 + tanh`: 71.96; unbounded spiking multiplicative gate after `o_proj`: 69.02 (1 seed, more seeds running) | E21 i1/i5, E20 |
| C4 | **avoid** | fighting overfitting by shrinking capacity (fewer gate dims / smaller projections) | the most popular agent hypothesis, refuted: 73.70 (2 seeds) | E21 i2 |
| C5 | **reflection** | judging an idea by its 2-epoch PPL | 2-epoch vs 30-epoch rank correlation is 0.25 over 12 architectures; among the best six (2-epoch spread 0.75 PPL, seed SD about 0.5) it carries no information. Do not claim a mechanism is better because of a sub-1-PPL 2-epoch gain; argue from what it does over long training (overfitting, gradient path, state stability) | E20, E18 |
| C6 | **gate** | shape bugs that only show at the real sequence length | also run `python scripts/cpu_check.py child.py --seq-len 127 --json check127.json`; it caught 7/7 shape failures with 0/40 false alarms | E24 |

When a card applies, prefer the validated pattern and write a hypothesis about **long-training** behaviour (C5), e.g.
"gate before `o_proj` keeps the residual path un-attenuated, so it should not overfit like C1".

3. 固定基准裁决：不退化，晋级（E27）。
4. 被提案者引用：E34 的 12 个提案中 9 个引用了卡片。按 OAF 选出的第 1 名 e34-rec-ffc28c 的假设原文：
   > Transplanting the identity (non-sigmoid) spiking multiplicative gate after o_proj from r02a-e29092 into the r04v2-bf4d6a DeltaNet, which uses elu_p1 q/k activation and no short-conv SiLU activation, will improve long-training performance (reduce validation PPL at 30 epochs) because the unbounded gate (per C3) can amplify or attenuate the residual path without the sigmoid's attenuation-only overfitting tendency (per C1), while the elu_p1 q/k activation should maintain the attention state quality.
5. 长训练裁决：30 轮 68.90（与冠军持平）。

## 案例 3：面板的成与败（E35）

- 4/5 个模型独立提出 SiLU 门，例如 Kimi K3 的理由：
   > Keeps the champion's unbounded positive amplification (for large positive g, g*sigmoid(g) ~ g, matching identity) while smoothly damping large negative g toward 0, adding a mild data-dependent regularizer that reduces variance from sign-flipping gates without capping the amplifying range that drives the champion's generalization.
   GPU：71.75（比冠军差 2.8）。
- GPT‑6.1 提出 soft‑clip：
   > Replace only the gate activation after o_proj; retain gate_proj and multiplication by o, with no new parameters. This odd, zero-centered soft clip preserves signed interactions and unit slope near zero while allowing amplification and compressing large magnitudes more gently than tanh. Conditional on useful interactions being moderate and extreme gains overfitting, it should slightly improve ident
   GPU：三种子 69.20（与冠军持平）。
- 各模型对 E33 三个变体的预测（2sig / tanh / abs；真值 74.14 / 70.16 / 69.70）：
  - dsv4flash: 71.5 / 70.5 / 69.5
  - dsv4pro: 71.9 / 71.5 / 69.5
  - glm53: 70.8 / 72.8 / 69.6
  - gpt61: 72.2 / 69.5 / 71.4
  - kimik3: 71.5 / 72.5 / 69.5

## 案例 4：Agent 盲设计择优标准（E32）

- GPT‑6.1 的假设原文：> At a fixed epoch-2 endpoint validation loss, a larger log(train2/val2) gap marks slower optimization and/or dropout robustness with more recoverable learning headroom. Subtracting this gap from endpoint validation loss should improve 30-epoch ranking on the held-out architectures; failure to beat val2 alone falsifies its predictive consequence.
- 公式：score = ln(val_ppl_ep2) - ln(train_ppl_ep2 / val_ppl_ep2) = 2*ln(val_ppl_ep2) - ln(train_ppl_ep2); lower is better. Equivalent ordering: val_ppl_ep2**2 / train_ppl_ep2. No fitted coefficients; code de；21 个前瞻测试架构上 ρ = 0.79（同子集 OAF 0.79，val₂ 0.06）。

## 案例 5：平台也当审稿人（E22，10/27）

- 5 个模型 × 2 次独立复核原始日志：10/10 复现主结论，8/10 数字完全一致；DeepSeek‑Flash 发现事件日志只存了表层报错（根因只在实验室日志）。E44 对本轮数据做同样的复核（进行中）。

## 案例 6：研究负责人（待补：从 E40‑L 挑一个 directives 原文与后续结果）
