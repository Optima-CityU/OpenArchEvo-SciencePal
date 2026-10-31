import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2norm
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

try:
    from src.models.spike.neuron import registry as neuron_registry
    from src.models.spike.surrogate import piecewise_quadratic_surrogate
except (ImportError, ModuleNotFoundError):
    class DummySurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input): return input
        @staticmethod
        def backward(ctx, grad_output): return grad_output
    neuron_registry = {'sdn': lambda surrogate_function: DummySurrogate.apply}
    piecewise_quadratic_surrogate = lambda: None

class SpikingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, neuron_type: str, surrogate_function):
        super().__init__()
        self.neuron = neuron_registry[neuron_type](surrogate_function=surrogate_function)
        self.linear = linear_layer

    def forward(self, x):
        x_transposed = x.transpose(-1, -2)
        spikes = self.neuron(x_transposed).transpose(-1, -2)
        return self.linear(spikes), spikes

# GSA white-box operator from i10-s0337

def gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=None):
    b, h, l, d = v.shape
    num_slots = q.shape[-1]
    
    pad_len = (chunk_size - l % chunk_size) % chunk_size
    if pad_len > 0:
        q, k, v, f = [F.pad(x, (0, 0, 0, pad_len)) for x in [q, k, v, f]]
        
    q = rearrange(q, 'b h (n c) m -> b h n c m', c=chunk_size)
    k = rearrange(k, 'b h (n c) m -> b h n c m', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    f = rearrange(f, 'b h (n c) m -> b h n c m', c=chunk_size)
    
    k = torch.softmax(k, dim=-1)
    q = torch.softmax(q, dim=-1)
    
    f = F.logsigmoid(f)
    
    if initial_state is not None:
        S = initial_state
    else:
        S = torch.zeros(b, h, num_slots, d, device=q.device)
        
    o_list = []
    
    for i in range(q.shape[2]):
        q_chunk = q[:, :, i]
        k_chunk = k[:, :, i]
        v_chunk = v[:, :, i]
        f_chunk = f[:, :, i]
        
        f_cumsum = torch.cumsum(f_chunk, dim=-2)
        
        decay_exp = torch.exp(f_cumsum)
        decay_inv_exp = torch.exp(-f_cumsum)
        
        q_mod = q_chunk * decay_exp
        k_mod = k_chunk * decay_inv_exp
        
        attn_score = q_mod @ k_mod.transpose(-1, -2)
        
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
        attn_score = attn_score.masked_fill(mask, 0)
        
        o_intra = attn_score @ v_chunk
        
        q_decayed = q_chunk * decay_exp
        o_inter = q_decayed @ S
        
        o_list.append(o_intra + o_inter)
        
        c_end_per_slot = f_cumsum[:, :, -1, :].unsqueeze(-2)
        decay_to_end_const_per_slot = torch.exp(c_end_per_slot)
        
        k_weighted_for_state_update = k_mod * decay_to_end_const_per_slot
        
        chunk_update = k_weighted_for_state_update.transpose(-1, -2) @ v_chunk
        
        current_chunk_total_decay_factor = torch.exp(f_chunk.sum(dim=-2)).unsqueeze(-1)
        
        S = S * current_chunk_total_decay_factor + chunk_update
        
    o = torch.stack(o_list, dim=2)
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    if pad_len > 0:
        o = o[..., :l, :]
    return o, S

class GSA(nn.Module):
    def __init__(self, d_model, num_heads, num_slots=64, expand_v=1.0, use_short_conv=True, conv_size=4, neuron_type='sdn', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.value_dim = int(d_model * expand_v)
        surrogate = piecewise_quadratic_surrogate()

        self.q_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.k_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.v_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)
        self.f_proj = SpikingLinear(nn.Linear(d_model, num_heads * num_slots, bias=False), neuron_type, surrogate)
        self.g_proj = SpikingLinear(nn.Linear(d_model, self.value_dim, bias=False), neuron_type, surrogate)

        self.o_proj = SpikingLinear(nn.Linear(self.value_dim, d_model, bias=False), neuron_type, surrogate)
        self.g_norm = RMSNorm(self.value_dim // num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
            self.f_conv1d = ShortConvolution(num_heads * num_slots, conv_size, activation='silu')

    def forward(self, x, **kwargs):
        all_spikes = {}

        q, s_q = self.q_proj(x)
        k, s_k = self.k_proj(x)
        v, s_v = self.v_proj(x)
        f, s_f = self.f_proj(x)
        g, s_g = self.g_proj(x)
        all_spikes.update({'q': s_q, 'k': s_k, 'v': s_v, 'f': s_f, 'g': s_g})

        if self.use_short_conv:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
            f, _ = self.f_conv1d(f)

        q = rearrange(q, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        k = rearrange(k, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        f = rearrange(f, 'b l (h m) -> b h l m', h=self.num_heads, m=self.num_slots)

        last_state = None
        o, final_state = gsa_chunkwise(q, k, v, f, chunk_size=64, initial_state=last_state)

        o = rearrange(o, 'b h l d -> b l h d')
        o = self.g_norm(o)

        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * F.silu(g)

        o = rearrange(o, 'b l h d -> b l (h d)')
        o, s_o = self.o_proj(o)
        all_spikes['o'] = s_o
        return o, None, all_spikes

class SpikingMLP(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, neuron_type='sdn', surrogate_function=None):
        super().__init__()
        self.fc1 = SpikingLinear(nn.Linear(d_model, d_ffn), neuron_type, surrogate_function)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = SpikingLinear(nn.Linear(d_ffn, d_model), neuron_type, surrogate_function)

    def forward(self, x):
        x, spike1 = self.fc1(x)
        x, spike2 = self.fc2(self.dropout(x))
        return x, spike1, spike2

class SpikingResidualGate(nn.Module):
    def __init__(self, d_model: int, neuron_type: str, surrogate_function):
        super().__init__()
        self.proj_in = SpikingLinear(nn.Linear(1, d_model // 2, bias=False), neuron_type, surrogate_function)
        self.proj_out = SpikingLinear(nn.Linear(d_model // 2, 1, bias=False), neuron_type, surrogate_function)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, avg_activity: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        spikes_list = []

        x_cont, spikes1 = self.proj_in(avg_activity)
        spikes_list.append(spikes1)

        x_act = F.gelu(x_cont)

        mod_factor_raw, spikes2 = self.proj_out(x_act)
        spikes_list.append(spikes2)

        modulation_factor = 1.0 + 2.0 * torch.sigmoid(mod_factor_raw + self.bias)

        return modulation_factor, spikes_list

class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult=2, dropout=0.1, prenorm=True, neuron_type='sdn', spike_reg_weight: float=0.01, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = RMSNorm(d_model)
        self.attn = GSA(d_model=d_model, num_heads=num_heads, neuron_type=neuron_type, **kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.attn_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * d_ffn_mult

        self.mlp = SpikingMLP(d_model=d_model, d_ffn=d_ffn, dropout=dropout, neuron_type=neuron_type, surrogate_function=piecewise_quadratic_surrogate())

        self.dropout2 = nn.Dropout(dropout)
        self.mlp_residual_gate = SpikingResidualGate(d_model, neuron_type, piecewise_quadratic_surrogate())

        self.spike_reg_weight = spike_reg_weight

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_spikes_list = []

        x_res = x
        if self.prenorm:
            x = self.norm1(x)

        x_attn, _, attn_spikes_dict = self.attn(x, **kwargs)

        attn_block_mean_spike_activities = []
        if attn_spikes_dict:
            for spike_tensor_name, spike_tensor in attn_spikes_dict.items():
                if spike_tensor is not None:
                    all_spikes_list.append(spike_tensor)
                    attn_block_mean_spike_activities.append(spike_tensor.float().mean(dim=-1, keepdim=True))

        if attn_block_mean_spike_activities:
            avg_attn_mean_activity = torch.mean(torch.stack(attn_block_mean_spike_activities), dim=0)
            attn_modulation_factor, attn_gate_spikes = self.attn_residual_gate(avg_attn_mean_activity)
            all_spikes_list.extend(attn_gate_spikes)
            x_attn = x_attn * attn_modulation_factor

        x = x_res + self.dropout1(x_attn)

        x_res = x
        if self.prenorm:
            x = self.norm2(x)

        x_mlp, mlp_spike1, mlp_spike2 = self.mlp(x)

        mlp_block_mean_spike_activities = []
        if mlp_spike1 is not None:
            all_spikes_list.append(mlp_spike1)
            mlp_block_mean_spike_activities.append(mlp_spike1.float().mean(dim=-1, keepdim=True))
        if mlp_spike2 is not None:
            all_spikes_list.append(mlp_spike2)
            mlp_block_mean_spike_activities.append(mlp_spike2.float().mean(dim=-1, keepdim=True))

        if mlp_block_mean_spike_activities:
            avg_mlp_mean_activity = torch.mean(torch.stack(mlp_block_mean_spike_activities), dim=0)
            mlp_modulation_factor, mlp_gate_spikes = self.mlp_residual_gate(avg_mlp_mean_activity)
            all_spikes_list.extend(mlp_gate_spikes)
            x_mlp = x_mlp * mlp_modulation_factor

        x = x_res + self.dropout2(x_mlp)

        final_spikes = torch.cat([s.flatten() for s in all_spikes_list]) if all_spikes_list else torch.empty(0, device=x.device, dtype=torch.bool)

        total_reg_loss = None
        if self.spike_reg_weight > 0 and final_spikes.numel() > 0:
            spike_activity_mean = torch.mean(final_spikes.float())
            total_reg_loss = spike_activity_mean * self.spike_reg_weight

        return x, total_reg_loss, final_spikes