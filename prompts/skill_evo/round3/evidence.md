# Evidence for revising the OpenArchEvo proposer Skill (v0.5.0)

v0.5.0 was written by an agent last round from the failures of v0.3.3 and won a fixed CPU benchmark. It was then used
inside the real GPU evolution loop (arm A5: 5 runs x 6 generations x 9 proposals, no research-lead agent).
Below is what happened to its proposals, next to the older arms.

# 提案去向（零 GPU）

| 组 | 运行数 | 提案 | CPU 检查未通过 | 近重复 | 行为与父代相同 | 功能等价门 | 进入种群 | 训练数（全部） | 选不出可训练候选的代数 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 5 | 702 | 103 | 147 | 91 | 0 | 353 | 156 | 2 |
| A | 5 | 720 | 50 | 242 | 280 | 0 | 148 | 132 | 12 |
| A5 | 5 | 270 | 23 | 60 | 30 | 0 | 152 | 65 | 0 |
| V2 | 5 | 720 | 50 | 155 | 154 | 2 | 361 | 151 | 3 |
| V2nk | 5 | 135 | 2 | 30 | 23 | 0 | 80 | 35 | 0 |

'CPU 检查未通过' = 被拒绝的原因为不可行 / 无候选 / CPU 检查失败的提案数（直接计数；10/25 23:10 修正，之前用减法会把进行中的提案算成失败）。'选不出可训练候选的代数' = 日志中 'nothing to select' 的次数。


## v0.5.0 in the GPU loop (A5): rejection reasons

- near_duplicate: 60
- parent_fingerprint_match: 30
- infeasible: 17
- fingerprint_failed: 4
- no_candidate: 2

'near_duplicate' = the child's behaviour fingerprint (loss, firing, output statistics on probe data) is almost the
same as a parent's, although the code changed. It is now the largest remaining source of waste.

## v0.5.0 in the GPU loop: CPU-check failures by stage (attempt level, most were repaired)

- forward: 64
- import: 24
- spiking_contract: 3
- causality: 2
- build: 2

## Examples of v0.5.0 proposals rejected as near-duplicates of a parent (real, from A5 runs)

### r01a5-b65a1d (role simplify, parent i00-mamba2-ssd): 3 changed non-comment lines; similarity to parents {"i00-mamba2-ssd": [0.984, 0.756, 1.0]}
```diff
-        self.dt_bias = nn.Parameter(torch.ones(num_heads) * -2.0)
-        a = -F.softplus(dt + self.dt_bias)
+        a = -F.softplus(dt - 1.0)
```

### r02a5-d604f1 (role recombine, parent r01a5-ca45bd): 223 changed non-comment lines; similarity to parents {"r01a5-ca45bd": [0.546, 0.916, 0.997], "i10-s0337": [0.919, 0.734, 0.249]}
```diff
-from einops import rearrange, repeat
+from einops import rearrange
+from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
-from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
+    print("Warning: Using placeholder SNN modules.")
+    """A sequence module containing LIF neurons and a linear layer."""
-    def forward(self, x):
-        x_transposed = x.transpose(-1, -2)
-        spikes = self.neuron(x_transposed).transpose(-1, -2)
-        return self.linear(spikes), spikes
-"""==================== White-box Operator (Mamba2 SSD) ===================="""
-def mamba2_ssd_parallel_stable(q, k, v, a, chunk_size=64, initial_state=None):
-    b, h, l, d = q.shape
+    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
+        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
+        spikes_transposed = self.neuron(x_transposed)
+        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D) # Binary spikes
+        output = self.linear(spikes) # The internal nn.Linear receives binary spikes
+        return output, spikes # output is analog, spikes are binary
+def gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=None):
+    """GSA Parallel Chunkwise Logic using Factorized Attention.
+    Significantly reduces memory by avoiding the materialization of the 5D decay mask.
+    Formula:
+        Score_ij = Sum_m (Q_im * exp(C_im)) * (K_jm * exp(-C_jm))
+    """
+    b, h, l, d = v.shape
+    num_slots = q.shape[-1]
-        q, k, v = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v]]
-        a = F.pad(a, (0, pad_len), value=0.0)
-    q, k, v = [rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size) for x in [q, k, v]]
```

### r02a5-52877b (role recombine, parent r01a5-42cb11): 275 changed non-comment lines; similarity to parents {"r01a5-42cb11": [0.54, 0.919, 0.998], "i10-s0337": [0.734, 0.697, 0.249]}
```diff
-def softmax(x):
-    return F.softmax(x, dim=-1)
-def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
-    b, h, l, d_k = q.shape
-    d_v = v.shape[-1]
+def gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=None):
+    """GSA Parallel Chunkwise Logic using Factorized Attention."""
+    b, h, l, d = v.shape
+    num_slots = q.shape[-1]
-        q = F.pad(q, (0, 0, 0, pad_len))
-        k = F.pad(k, (0, 0, 0, pad_len))
-        v = F.pad(v, (0, 0, 0, pad_len))
-        beta = F.pad(beta, (0, pad_len))
-    padded_len = l + pad_len
-    q = l2norm(q)
-    k = l2norm(k)
-    v = v * beta[..., None]
-    k_beta = k * beta[..., None]
-    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
-    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
-    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
-    for i in range(1, chunk_size):
-        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
-    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
-    u = attn @ v
-    w = attn @ k_beta
-    S = k.new_zeros(b, h, d_k, d_v)
-    o = torch.zeros_like(v)
-    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
-    for i in range(0, padded_len // chunk_size):
```

### r03a5-9a6ebd (role simplify, parent r02a5-cbd096): 23 changed non-comment lines; similarity to parents {"r02a5-cbd096": [0.937, 0.911, 0.245]}
```diff
-class SpikingResidualGate(nn.Module):
-    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
-        super().__init__()
-        self.proj_in = SpikingLinear(nn.Linear(1, d_model // 2, bias=False), neuron_type, surrogate_function)
-        self.proj_out = SpikingLinear(nn.Linear(d_model // 2, 1, bias=False), neuron_type, surrogate_function)
-        self.bias = nn.Parameter(torch.zeros(1))
-    def forward(self, avg_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
-        spikes_list = []
-        x_cont, spikes1 = self.proj_in(avg_activity)
-        spikes_list.append(spikes1)
-        x_act = F.gelu(x_cont)
-        mod_factor_raw, spikes2 = self.proj_out(x_act)
-        spikes_list.append(spikes2)
-        modulation_factor = 1.0 + 2.0 * torch.sigmoid(mod_factor_raw + self.bias)
-        return modulation_factor, spikes_list
-        self.attn_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())
-        attn_block_mean_spike_activities = []
-                    attn_block_mean_spike_activities.append(spike_tensor.float().mean(dim=-1, keepdim=True))
-        if attn_block_mean_spike_activities:
-            avg_attn_mean_activity = torch.mean(torch.stack(attn_block_mean_spike_activities), dim=0)
-            attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(avg_attn_mean_activity)
-            all_spikes_list.extend(attn_gate_spikes)
-            x_attn = x_attn * attn_modulation_factor
```

### r04a5-f93748 (role simplify, parent r01a5-654db4): 23 changed non-comment lines; similarity to parents {"r01a5-654db4": [0.94, 0.906, 0.249]}
```diff
-class SpikingResidualGate(nn.Module):
-    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
-        super().__init__()
-        self.proj_in = SpikingLinear(nn.Linear(1, d_model // 2, bias=False), neuron_type, surrogate_function)
-        self.proj_out = SpikingLinear(nn.Linear(d_model // 2, 1, bias=False), neuron_type, surrogate_function)
-        self.bias = nn.Parameter(torch.zeros(1))
-    def forward(self, avg_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
-        spikes_list = []
-        x_cont, spikes1 = self.proj_in(avg_activity)
-        spikes_list.append(spikes1)
-        x_act = F.gelu(x_cont)
-        mod_factor_raw, spikes2 = self.proj_out(x_act)
-        spikes_list.append(spikes2)
-        modulation_factor = 1.0 + 2.0 * torch.sigmoid(mod_factor_raw + self.bias)
-        return modulation_factor, spikes_list
-        self.attn_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())
-        attn_block_mean_spike_activities = []
-                    attn_block_mean_spike_activities.append(spike_tensor.float().mean(dim=-1, keepdim=True))
-        if attn_block_mean_spike_activities:
-            avg_attn_mean_activity = torch.mean(torch.stack(attn_block_mean_spike_activities), dim=0)
-            attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(avg_attn_mean_activity)
-            all_spikes_list.extend(attn_gate_spikes)
-            x_attn = x_attn * attn_modulation_factor
```

## Examples of v0.5.0 proposals rejected because they behave exactly like their parent (real, from A5 runs)

### r04a5-b2a0eb (role explore, parent r03a5-21e348): 4 changed non-comment lines; similarity to parents {"r03a5-21e348": [0.999, 0.96, 1.0]}
```diff
-        self.f_proj2 = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
+        self.f2_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
-        f2, s_f2 = self.f_proj2(x)
+        f2, s_f2 = self.f2_proj(x)
```

### r05a5-a88047 (role simplify, parent r03a5-ce5c74): 28 changed non-comment lines; similarity to parents {"r03a5-ce5c74": [0.965, 0.927, 1.0]}
```diff
-        self.use_gate = use_gate
+        self.use_gate = False
-        self.use_spike_gate_modulation = use_spike_gate_modulation
-        if use_gate:
-            self.g_proj = SpikingLinear(nn.Linear(hidden_size, self.value_dim, bias=False), neuron_type, surrogate_function)
-            if self.use_spike_gate_modulation:
-                self.spike_gate_mod_proj = SpikingLinear(nn.Linear(3, self.head_v_dim * self.num_heads, bias=True), neuron_type, surrogate_function)
-            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
-        else:
-            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
+        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
-        if (self.use_beta and self.use_spike_beta_modulation) or (self.use_gate and self.use_spike_gate_modulation):
+        if self.use_beta and self.use_spike_beta_modulation:
-        if self.use_gate:
-            g, spikes_g = self.g_proj(hidden_states)
-            all_spikes['g_proj'] = spikes_g
-            if self.use_spike_gate_modulation and aggregated_spikes is not None:
-                spike_gate_mod_raw, spikes_gate_mod = self.spike_gate_mod_proj(aggregated_spikes)
-                gate_modulation = torch.sigmoid(spike_gate_mod_raw)
-                g = g * (1 + gate_modulation)
-                all_spikes['spike_gate_mod'] = spikes_gate_mod
-            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
-            o = self.o_norm(o, g)
-        else:
-            o = self.o_norm(o)
+        o = self.o_norm(o)
-        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, use_spike_beta_modulation=use_spike_beta_modulation, use_spike_gate_modulation=use_spike_gate_modulation, **kwargs)
+        self.attn = DeltaNet(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, use_spike_beta_modulation=use_spike_beta_modulation, **kwargs)
```

## Other facts

- The search scores children by 2-epoch WikiText-2 perplexity. A 30-epoch check showed that the best 2/6-epoch
  finding (a spiking output gate on DeltaNet) learns fast early but ends 1.7 PPL worse than DeltaNet at 30 epochs,
  while a parameter-free change (elu+1 feature map on q/k) was 1.1 PPL better at 30 epochs. Children that only add
  capacity or speed up early learning can look good to the proxy without being better.
- In the fixed benchmark, effective new proposals (accepted and not equivalent to the parent) are already 43/44 for
  v0.5.0. The open problem is that repeated requests with the same parents and role tend to yield the same idea,
  and in the GPU loop many children stay behaviourally too close to their parent.
- Last rounds: an agent-written v0.4 was rejected (longer, more repairs); v0.5.0 won by removing parent-equivalent
  children without lowering the CPU pass rate. Longer is not better.
