# Evidence for revising the OpenArchEvo proposer Skill (v0.3.3)

You are improving the instructions that a DeepSeek proposer model follows when it writes one child architecture
program per request (roles explore / recombine / simplify). Below is what happened to its proposals this week.

# 提案去向（零 GPU）

| 组 | 运行数 | 提案 | CPU 检查未通过 | 近重复 | 行为与父代相同 | 功能等价门 | 进入种群 | 训练数（全部） | 选不出可训练候选的代数 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 5 | 342 | 44 | 74 | 43 | 0 | 181 | 81 | 0 |
| A | 5 | 360 | 29 | 113 | 102 | 0 | 116 | 78 | 3 |
| V2 | 5 | 360 | 22 | 73 | 73 | 1 | 191 | 85 | 0 |

'CPU 检查未通过' = 提案数减去其余各项（修复 2 次后仍失败或解析失败）。'选不出可训练候选的代数' = 日志中 'nothing to select' 的次数。


## Where proposals failed the CPU check (stage counts, Skill v0.3.3 runs)

- forward: 119
- import: 69
- spiking_contract: 13
- causality: 7
- build: 5

## Examples of proposals rejected because they behave exactly like their parent (real, from Skill v0.3.3 runs)

### r02a-c5fe99 (role recombine, parent i00-deltanet): 5 changed non-comment lines
```diff
-        if use_gate:
+        self.use_gate = use_gate
+        if self.use_gate:
-        all_spikes = {} # Used to collect all spikes
+        all_spikes = {}
```

### r02a-a27ae0 (role simplify, parent i00-deltanet): 3 changed non-comment lines
```diff
-        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
+        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)
-        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)
```

### r04a-b7742d (role explore, parent i00-deltanet): 13 changed non-comment lines
```diff
-    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> DeltaNet:
+    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, num_groups: int=1, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> DeltaNet:
+        self.num_groups = num_groups
+        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
+        self.num_query_heads = num_heads // num_groups
+        self.query_head_dim = self.key_dim // self.num_query_heads
-        all_spikes = {} # Used to collect all spikes
+        all_spikes = {}
-        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
+        q = rearrange(q, '... (h d) -> ... h d', d=self.query_head_dim)
+        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
+        q = q.repeat_interleave(self.num_groups, dim=1)
-            print(f"INFO: Layer uses SpikingMLP with {neuron_type} neuron.")
```

### r05a-7da371 (role recombine, parent i00-deltanet): 48 changed non-comment lines
```diff
-from __future__ import annotations
-from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
-from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
-from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
+from fla.modules import RMSNorm, ShortConvolution
-    """A sequence module containing LIF neurons and a linear layer."""
-    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
-        x_transposed = x.transpose(-1, -2)  # (B, L, D) -> (B, D, L)
+    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
+        x_transposed = x.transpose(-1, -2)
-        spikes = spikes_transposed.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
+        spikes = spikes_transposed.transpose(-1, -2)
-def softmax(x):
-    return F.softmax(x, dim=-1)
-if TYPE_CHECKING:
-    from transformers.processing_utils import Unpack
-    from fla.models.utils import Cache
-def elu_p1(x):
-    return (F.elu(x, 1., False) + 1.).to(x)
-def sum_norm(x):
-    return (x / x.sum(-1, keepdim=True)).to(x)
-    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='sdn', **kwargs) -> DeltaNet:
+    def __init__(self, mode: str='chunk1', d_model: int=None, hidden_size: int=1024, expand_k: float=1.0, expand_v: float=1.0, num_heads: int=4, use_beta: bool=True, use_gate: bool=False, use_short_conv: bool=True, conv_size: int=4, conv_bias: bool=False, allow_neg_eigval: bool=False, layer_idx: int=None, qk_activation: str='silu', qk_norm: str='l2', norm_eps: float=1e-05, neuron_type: str='lif', **kwargs):
-            raise ImportError("Spiking neuron modules are not available. Please check your installation.")
+            raise ImportError("Spiking neuron modules are not available.")
-        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
-        assert self.qk_norm in ['l2', 'sum']
-        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {{num_heads}}"
-        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {{num_heads}}"
+        assert self.key_dim % num_heads == 0
```

## Other facts

- Without per-slot research directions (arm A), 60% of v0.3.3 proposals were near-duplicates or behaved like their parent,
  and one run had nothing new to train for 3 generations. With directions from a research-lead agent (V2): 41%.
- Many 'simplify' children change parameters slightly but compute the same function (dead code, re-parameterisation).
- The best mechanism found this week (spiking output gate on DeltaNet, confirmed with 3 seeds and a parameter-matched
  control) came from 'explore'/'recombine' slots with a new live computation path.
- A Skill candidate written by an agent last week (v0.4) claimed improvements in its own back-test but did worse on the
  fixed benchmark (more repairs, lower first-try pass) and was rejected: longer is not better.
