import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from utils.config import cfg

if __name__ == '__main__':
    from utils import init_param, Scaler  # When debug use this code
else:
    from .utils import init_param, Scaler  # When run main use it

class TCBMaskedParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, threshold):
        threshold = threshold.to(device=param.device, dtype=param.dtype)
        mask = torch.abs(param) >= threshold
        ctx.save_for_backward(param, threshold, mask)
        return param * mask.to(dtype=param.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        param, threshold, mask = ctx.saved_tensors
        scores = torch.abs(param)
        bias = 1.0 + (2 * scores * threshold / (torch.pow(scores + threshold, 2) + 1e-8))
        bias = torch.nan_to_num(bias, nan=1.0, posinf=1.0, neginf=1.0)
        return grad_output * mask.to(dtype=grad_output.dtype) * bias, None


# Shared mixin for masked layers; provides mask state and logic rather than a complete network layer.
class MaskedMixin:
    def _init_mask_state(self):
        self._mask_thresholds = {}  # Store the threshold for each parameter.
        self._mask_enabled = {}     # Track whether masking is enabled for each parameter.
        self._mask_bern = {}
        self._fixed_masks = {}

    def set_threshold(self, value, bern=True, param_name='weight', fixed_mask=None):
        if not hasattr(self, param_name):
            raise ValueError(f'Unsupported parameter name: {param_name}')
        param = getattr(self, param_name)
        if param is None:
            raise ValueError(f'Parameter {param_name} is None')
        self._mask_thresholds[param_name] = value.detach().clone()
        self._mask_enabled[param_name] = True
        self._mask_bern[param_name] = bern
        if fixed_mask is None:
            self._fixed_masks.pop(param_name, None)     # Return None without raising an error if the key is absent.
        else:
            self._fixed_masks[param_name] = fixed_mask.detach().clone()

    def clear_threshold(self):
        self._mask_thresholds.clear()
        self._mask_enabled.clear()
        self._mask_bern.clear()
        self._fixed_masks.clear()

    def _masked_param(self, param_name):
        # Return the effective version of the current parameter.
        param = getattr(self, param_name)
        if param is None or not self._mask_enabled.get(param_name, False):
            return param
        threshold = self._mask_thresholds[param_name].to(device=param.device, dtype=param.dtype)
        scores = torch.abs(param)
        if param_name in self._fixed_masks:
            mask = self._fixed_masks[param_name].to(device=param.device, dtype=param.dtype)
            return param * mask
        elif self._mask_bern.get(param_name, True):
            return TCBMaskedParam.apply(param, threshold)
        else:
            mask = scores >= threshold
        return param * mask


class MaskedEmbedding(MaskedMixin, nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mask_state()

    def forward(self, input):
        return F.embedding(
            input,
            self._masked_param('weight'),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class MaskedLinear(MaskedMixin, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mask_state()

    def forward(self, input):
        return F.linear(input, self._masked_param('weight'), self._masked_param('bias'))


class MaskedLayerNorm(MaskedMixin, nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mask_state()

    def forward(self, input):
        return F.layer_norm(
            input,
            self.normalized_shape,
            self._masked_param('weight'),
            self._masked_param('bias'),
            self.eps,
        )


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        # cfg['bptt'] specifies each input sequence length during language-model training.
        self.positional_embedding = MaskedEmbedding(cfg['bptt'], embedding_size)

    def forward(self, x):
        N, S = x.size()     # [batch_size, seq_len]
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout, rate):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = MaskedEmbedding(num_tokens, embedding_size)
        self.norm = MaskedLayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.scaler = Scaler(rate)

    def forward(self, src):
        src = self.scaler(self.embedding(src)) + self.scaler(self.positional_embedding(src))
        src = self.dropout(self.norm(src))
        return src


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # .transpose(-2, -1) swaps the last two dimensions and leaves the others unchanged.
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        # When a mask is provided, set hidden positions to -inf so their softmax probabilities are approximately zero.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)    # Apply softmax along the last dimension.
        output = torch.matmul(attn, v)
        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, rate):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.linear_q = MaskedLinear(embedding_size, embedding_size)
        self.linear_k = MaskedLinear(embedding_size, embedding_size)
        self.linear_v = MaskedLinear(embedding_size, embedding_size)
        self.linear_o = MaskedLinear(embedding_size, embedding_size)
        head_dim = embedding_size // num_heads
        self.attention = ScaledDotProduct(temperature=head_dim ** 0.5)
        self.scaler = Scaler(rate)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.scaler(self.linear_q(q)), self.scaler(self.linear_k(k)), self.scaler(self.linear_v(v))
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.scaler(self.linear_o(q))
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads, rate=rate)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = MaskedLayerNorm(embedding_size)
        self.linear1 = MaskedLinear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = MaskedLinear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = MaskedLayerNorm(embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.init_param()

    def init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        return

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src_key_padding_mask marks padding tokens so attention ignores those positions.
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.scaler(self.linear2(self.dropout1(self.activation(self.scaler(self.linear1(src))))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, rate):
        super().__init__()
        self.linear1 = MaskedLinear(embedding_size, embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.norm1 = MaskedLayerNorm(embedding_size)
        self.linear2 = MaskedLinear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.scaler(self.linear1(src)))))
        return out


class Transformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, rate):
        super().__init__()
        self.idx = OrderedDict()
        self.thresholds = OrderedDict()
        self.roll = 0
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if embedding_size % num_heads != 0:
            raise ValueError(
                f'embedding_size ({embedding_size}) must be divisible by num_heads ({num_heads})'
            )
        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout, rate)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = Decoder(num_tokens, embedding_size, rate)

    def _num_keep(self, size, rate):
        return max(1, min(size, int(rate * size)))

    def _top_index(self, score, k, select_mode='l2'):
        score = score.detach().float().cpu()
        k = max(1, min(k, score.numel()))
        if select_mode == 'absmin':
            return torch.sort(torch.topk(-score.abs(), k).indices)[0]
        if select_mode == 'probs':
            prob = score.abs()
            total = prob.sum()
            if total <= 0:
                return torch.sort(torch.randperm(score.numel())[:k])[0]
            return torch.sort(torch.multinomial(prob / total, k, replacement=False))[0]
        return torch.sort(torch.topk(score.abs(), k).indices)[0]

    def _head_dim(self, size):
        if size % self.num_heads != 0:
            raise ValueError(
                f'Head-aware selection requires size ({size}) divisible by num_heads ({self.num_heads})'
            )
        return size // self.num_heads

    def _head_keep(self, size, rate):
        k = self._num_keep(size, rate)
        if k % self.num_heads != 0:
            raise ValueError(
                f'Head-aware selection requires kept channels ({k}) divisible by num_heads ({self.num_heads})'
            )
        return k // self.num_heads

    def _headwise_index(self, size, rate, select_fn):
        head_dim = self._head_dim(size)
        keep_per_head = self._head_keep(size, rate)
        selected = []
        for head in range(self.num_heads):
            start = head * head_dim
            local_idx = select_fn(head, head_dim, keep_per_head)
            selected.append(local_idx + start)
        return torch.sort(torch.cat(selected))[0]

    def _headwise_top_index(self, score, rate, select_mode='l2'):
        score = score.detach().float().cpu()

        def select_fn(head, head_dim, keep_per_head):
            start = head * head_dim
            return self._top_index(score[start:start + head_dim], keep_per_head, select_mode)

        return self._headwise_index(score.numel(), rate, select_fn)

    def _headwise_roll_index(self, size, rate, roll_offset=None):
        roll = self.roll if roll_offset is None else roll_offset

        def select_fn(head, head_dim, keep_per_head):
            return torch.sort(torch.roll(torch.arange(head_dim), shifts=roll % head_dim, dims=0)[:keep_per_head])[0]

        return self._headwise_index(size, rate, select_fn)

    def _headwise_rand_index(self, size, rate):
        return self._headwise_index(size, rate, lambda head, head_dim, keep_per_head: torch.randperm(head_dim)[:keep_per_head])

    def _headwise_hetero_index(self, size, rate):
        return self._headwise_index(size, rate, lambda head, head_dim, keep_per_head: torch.arange(keep_per_head))

    def _headwise_ecover_index(self, name, size, rate):
        return self._headwise_index(
            size,
            rate,
            lambda head, head_dim, keep_per_head: self._ecover_index(f'{name}.head{head}', head_dim, keep_per_head / head_dim),
        )

    def _roll_index(self, size, rate, roll_offset=None):
        roll = self.roll if roll_offset is None else roll_offset
        return torch.roll(torch.arange(size), shifts=roll % size, dims=0)[:self._num_keep(size, rate)]

    def _rand_index(self, size, rate):
        return torch.randperm(size)[:self._num_keep(size, rate)]

    def _hetero_index(self, size, rate):
        return torch.arange(self._num_keep(size, rate))

    def _score_along_dim(self, tensor, dim, select_mode='l2'):
        score = tensor.detach().abs().float().cpu()
        reduce_dims = tuple(i for i in range(score.dim()) if i != dim)
        if not reduce_dims:
            return score
        if select_mode == 'l2':
            return score.pow(2).sum(dim=reduce_dims)
        return score.sum(dim=reduce_dims)

    def _param_score(self, gradient, name, dim, size, select_mode='l2'):
        if gradient is None or name not in gradient:
            return None
        score = self._score_along_dim(gradient[name], dim, select_mode)
        if score.numel() != size:
            raise ValueError(
                f'Gradient score size mismatch for {name} dim {dim}: '
                f'expected {size}, got {score.numel()}'
            )
        return score

    def _normalize_param_score(self, score):
        denom = score.abs().mean()
        if denom <= 1e-12:
            return score
        return score / (denom + 1e-12)

    def _mean_param_scores(self, gradient, sources, size, select_mode='l2'):
        total = torch.zeros(size)
        count = 0
        for name, dim in sources:
            score = self._param_score(gradient, name, dim, size, select_mode)
            if score is None:
                continue
            score = self._normalize_param_score(score)
            total += score
            count += 1
        if count == 0:
            return torch.ones(size)
        return total / count

    def _ecover_index(self, name, size, rate):
        if not hasattr(self, '_ecover_buffer'):
            self._ecover_buffer = {}
        if name not in self._ecover_buffer:
            self._ecover_buffer[name] = torch.arange(size)
        k = self._num_keep(size, rate)
        buffer = self._ecover_buffer[name]
        if k > buffer.numel():
            first = buffer.clone()
            remain = k - first.numel()
            pool = torch.arange(size)
            available = pool[~torch.isin(pool, first)]
            extra = available[torch.randperm(available.numel())[:remain]]
            selected = torch.sort(torch.cat([first, extra]))[0]
            self._ecover_buffer[name] = pool[~torch.isin(pool, extra)]
        else:
            selected = torch.sort(buffer[torch.randperm(buffer.numel())[:k]])[0]
            self._ecover_buffer[name] = buffer[~torch.isin(buffer, selected)]
        return selected

    def _aware_indices(self, input, rate, select_mode):
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                src = input['label'][:, :-1].contiguous()
                seq_len = src.size(1)
                attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool))
                x = self.transformer_embedding(src)
                embed_score = x.detach().abs().float().sum(dim=(0, 1)).cpu()
                hidden_scores = []
                for layer in self.transformer_encoder.layers:
                    attn_output, _ = layer.mha(x, x, x, mask=attn_mask)
                    x = layer.norm1(x + layer.dropout(attn_output))
                    hidden = layer.activation(layer.scaler(layer.linear1(x)))
                    hidden_scores.append(hidden.detach().abs().float().sum(dim=(0, 1)).cpu())
                    x = layer.norm2(x + layer.dropout2(layer.scaler(layer.linear2(layer.dropout1(hidden)))))
        finally:
            self.train(was_training)
        embed_idx = self._headwise_top_index(embed_score, rate, select_mode)
        hidden_idx = [
            self._top_index(score, self._num_keep(self.hidden_size, rate), select_mode)
            for score in hidden_scores
        ]
        return embed_idx, hidden_idx

    def _gradient_indices(self, rate, select_mode, gradient):
        embed_sources = [
            ('transformer_embedding.embedding.weight', 1),
            ('transformer_embedding.positional_embedding.positional_embedding.weight', 1),
            ('transformer_embedding.norm.weight', 0),
            ('transformer_embedding.norm.bias', 0),
            ('decoder.linear1.weight', 0),
            ('decoder.linear1.weight', 1),
            ('decoder.linear1.bias', 0),
            ('decoder.norm1.weight', 0),
            ('decoder.norm1.bias', 0),
            ('decoder.linear2.weight', 1),
        ]
        for i in range(self.num_layers):
            prefix = f'transformer_encoder.layers.{i}'
            for name in ['linear_q', 'linear_k', 'linear_v', 'linear_o']:
                embed_sources.extend([
                    (f'{prefix}.mha.{name}.weight', 0),
                    (f'{prefix}.mha.{name}.weight', 1),
                    (f'{prefix}.mha.{name}.bias', 0),
                ])
            embed_sources.extend([
                (f'{prefix}.norm1.weight', 0),
                (f'{prefix}.norm1.bias', 0),
                (f'{prefix}.linear1.weight', 1),
                (f'{prefix}.linear2.weight', 0),
                (f'{prefix}.linear2.bias', 0),
                (f'{prefix}.norm2.weight', 0),
                (f'{prefix}.norm2.bias', 0),
            ])
        embed_score = self._mean_param_scores(gradient, embed_sources, self.embedding_size, select_mode)
        embed_idx = self._headwise_top_index(embed_score, rate, select_mode)

        hidden_idx = []
        for i in range(self.num_layers):
            prefix = f'transformer_encoder.layers.{i}'
            hidden_sources = [
                (f'{prefix}.linear1.weight', 0),
                (f'{prefix}.linear1.bias', 0),
                (f'{prefix}.linear2.weight', 1),
            ]
            score = self._mean_param_scores(gradient, hidden_sources, self.hidden_size, select_mode)
            hidden_idx.append(self._top_index(score, self._num_keep(self.hidden_size, rate), select_mode))
        return embed_idx, hidden_idx

    def _build_idx(self, embed_idx, hidden_indices):
        token_idx = torch.arange(self.num_tokens)
        pos_idx = torch.arange(cfg['bptt'])
        self.idx.clear()

        self.idx['transformer_embedding.positional_embedding.positional_embedding.weight'] = (pos_idx, embed_idx)
        self.idx['transformer_embedding.embedding.weight'] = (token_idx, embed_idx)
        self.idx['transformer_embedding.norm.weight'] = embed_idx
        self.idx['transformer_embedding.norm.bias'] = embed_idx

        for i in range(self.num_layers):
            prefix = f'transformer_encoder.layers.{i}'
            hidden_idx = hidden_indices[i]
            for name in ['linear_q', 'linear_k', 'linear_v', 'linear_o']:
                self.idx[f'{prefix}.mha.{name}.weight'] = (embed_idx, embed_idx)
                self.idx[f'{prefix}.mha.{name}.bias'] = embed_idx
            self.idx[f'{prefix}.norm1.weight'] = embed_idx
            self.idx[f'{prefix}.norm1.bias'] = embed_idx
            self.idx[f'{prefix}.linear1.weight'] = (hidden_idx, embed_idx)
            self.idx[f'{prefix}.linear1.bias'] = hidden_idx
            self.idx[f'{prefix}.linear2.weight'] = (embed_idx, hidden_idx)
            self.idx[f'{prefix}.linear2.bias'] = embed_idx
            self.idx[f'{prefix}.norm2.weight'] = embed_idx
            self.idx[f'{prefix}.norm2.bias'] = embed_idx

        self.idx['decoder.linear1.weight'] = (embed_idx, embed_idx)
        self.idx['decoder.linear1.bias'] = embed_idx
        self.idx['decoder.norm1.weight'] = embed_idx
        self.idx['decoder.norm1.bias'] = embed_idx
        self.idx['decoder.linear2.weight'] = (token_idx, embed_idx)
        self.idx['decoder.linear2.bias'] = token_idx

    def get_idx_rand(self, rate):
        embed_idx = self._headwise_rand_index(self.embedding_size, rate)
        hidden_idx = [self._rand_index(self.hidden_size, rate) for _ in range(self.num_layers)]
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_rand_grad(self, rate, select_mode, gradient, epoch=None):
        embed_idx, hidden_idx = self._gradient_indices(rate, select_mode, gradient)
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_roll(self, rate, roll_offset=None):
        embed_idx = self._headwise_roll_index(self.embedding_size, rate, roll_offset)
        hidden_idx = [self._roll_index(self.hidden_size, rate, roll_offset) for _ in range(self.num_layers)]
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_ecover(self, rate, model_name=None):
        embed_idx = self._headwise_ecover_index('embedding', self.embedding_size, rate)
        hidden_idx = [
            self._ecover_index(f'transformer_encoder.layers.{i}.linear1.weight', self.hidden_size, rate)
            for i in range(self.num_layers)
        ]
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_hetero(self, rate):
        embed_idx = self._headwise_hetero_index(self.embedding_size, rate)
        hidden_idx = [self._hetero_index(self.hidden_size, rate) for _ in range(self.num_layers)]
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_aware(self, input, rate, select_mode):
        embed_idx, hidden_idx = self._aware_indices(input, rate, select_mode)
        self._build_idx(embed_idx, hidden_idx)

    def get_idx_aware_grad(self, rate, select_mode, gradient):
        embed_idx, hidden_idx = self._gradient_indices(rate, select_mode, gradient)
        self._build_idx(embed_idx, hidden_idx)

    def get_mask_fiarse(self, rate, use_tcbgd=None):
        if use_tcbgd is None:
            use_tcbgd = cfg.get('fiarse_use_tcbgd', True)
        self.thresholds.clear()
        for module_name, module in self.named_modules():
            params = [(name, param) for name, param in module.named_parameters(recurse=False)
                      if torch.is_floating_point(param)]
            if not params:
                continue
            # Direct parameters of a layer share a threshold, including weight and bias.
            scores = torch.cat([param.detach().abs().reshape(-1) for _, param in params])
            k = max(1, min(scores.numel(), int(scores.numel() * rate)))
            threshold = torch.topk(scores, k=k).values[-1]
            for tensor_name, param in params:
                if hasattr(module, 'set_threshold'):
                    fixed_mask = None if use_tcbgd else param.detach().abs() >= threshold
                    module.set_threshold(threshold, bern=use_tcbgd,
                                         param_name=tensor_name, fixed_mask=fixed_mask)
                name = f'{module_name}.{tensor_name}' if module_name else tensor_name
                self.thresholds[name] = threshold

    def clear_idx(self):
        self.idx.clear()
        self.thresholds.clear()
        for module in self.modules():
            if hasattr(module, 'clear_threshold'):
                module.clear_threshold()

    def forward(self, input):
        output = {}
        tokens = input['label']
        src = tokens[:, :-1].contiguous()
        target = tokens[:, 1:].contiguous()
        seq_len = src.size(1)
        # Construct a lower-triangular causal mask.
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool))
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src, mask=attn_mask)
        out = self.decoder(src)
        out = out.permute(0, 2, 1)  # out shape: [batch_size, num_tokens, seq_len - 1]
        if 'label_split' in input and cfg.get('mask', False):
            label_mask = torch.zeros((cfg['num_tokens'], 1), device=out.device)
            label_mask[input['label_split']] = 1
            out = out.masked_fill(label_mask == 0, -1e9)
        output['score'] = out
        output['loss'] = F.cross_entropy(output['score'], target)
        return output


def transformer(model_rate=1):
    num_tokens = cfg['num_tokens']
    embedding_size = int(np.ceil(model_rate * cfg['transformer']['embedding_size']))
    num_heads = cfg['transformer']['num_heads']
    hidden_size = int(np.ceil(model_rate * cfg['transformer']['hidden_size']))
    num_layers = cfg['transformer']['num_layers']
    dropout = cfg['transformer']['dropout']
    scaler_rate = model_rate
    model = Transformer(num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, scaler_rate)
    model.apply(init_param)
    return model
