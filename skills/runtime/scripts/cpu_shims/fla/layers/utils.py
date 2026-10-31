import torch
import torch.nn.functional as F


def get_unpad_data(attention_mask):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_len = int(seqlens.max().item())
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu, max_len


def index_first_axis(x, indices):
    return x[indices]


def pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros((batch * seqlen,) + tuple(hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])
