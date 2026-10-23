import math

import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
from utils.config import cfg

if __name__ == '__main__':
    from utils import init_param, Scaler
else:
    from .utils import init_param, Scaler


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

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_features: int, out_features: int, kernel_size, **kwargs):
        super().__init__(in_features, out_features, kernel_size=kernel_size, **kwargs)
        self.weight_threshold = torch.tensor(0.)
        self.bias_threshold = torch.tensor(0.)
        self.weight_fixed_mask = None
        self.bias_fixed_mask = None
        self.bern = True  # Apply Bern() function

    @property
    def threshold(self):
        return self.weight_threshold

    def forward(self, x):
        weight_threshold = self.weight_threshold.to(device=self.weight.device, dtype=self.weight.dtype)
        if self.weight_fixed_mask is not None:
            weight_mask = self.weight_fixed_mask.to(device=self.weight.device, dtype=self.weight.dtype)
            effective_weight = self.weight * weight_mask
        elif self.bern:
            effective_weight = TCBMaskedParam.apply(self.weight, weight_threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= weight_threshold)
            effective_weight = self.weight * weight_mask
        if self.bias is not None:
            bias_threshold = self.bias_threshold.to(device=self.bias.device, dtype=self.bias.dtype)
            if self.bias_fixed_mask is not None:
                bias_mask = self.bias_fixed_mask.to(device=self.bias.device, dtype=self.bias.dtype)
                effective_bias = self.bias * bias_mask
            elif self.bern:
                effective_bias = TCBMaskedParam.apply(self.bias, bias_threshold)
            else:
                bias_mask = (torch.abs(self.bias) >= bias_threshold)
                effective_bias = self.bias * bias_mask
        else:
            effective_bias = None
        return self._conv_forward(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True, param_name='weight', fixed_mask=None):
        if param_name == 'weight':
            self.weight_threshold = value
            self.weight_fixed_mask = fixed_mask.detach().clone() if fixed_mask is not None else None
        elif param_name == 'bias':
            self.bias_threshold = value
            self.bias_fixed_mask = fixed_mask.detach().clone() if fixed_mask is not None else None
        else:
            raise ValueError(f'Unsupported parameter name: {param_name}')
        self.bern = bern

    def clear_threshold(self):
        self.weight_threshold = torch.tensor(0.)
        self.bias_threshold = torch.tensor(0.)
        self.weight_fixed_mask = None
        self.bias_fixed_mask = None
        self.bern = True

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
            + (np.prod(self.bias.shape) if self.bias is not None else 0)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        self.weight_threshold = torch.tensor(0.)
        self.bias_threshold = torch.tensor(0.)
        self.weight_fixed_mask = None
        self.bias_fixed_mask = None
        self.bern = True  # Apply Bern() function

    @property
    def threshold(self):
        return self.weight_threshold

    def forward(self, x):
        weight_threshold = self.weight_threshold.to(device=self.weight.device, dtype=self.weight.dtype)
        if self.weight_fixed_mask is not None:
            weight_mask = self.weight_fixed_mask.to(device=self.weight.device, dtype=self.weight.dtype)
            effective_weight = self.weight * weight_mask
        elif self.bern:
            effective_weight = TCBMaskedParam.apply(self.weight, weight_threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= weight_threshold)
            effective_weight = self.weight * weight_mask

        if self.bias is None:
            effective_bias = None
        else:
            bias_threshold = self.bias_threshold.to(device=self.bias.device, dtype=self.bias.dtype)
            if self.bias_fixed_mask is not None:
                bias_mask = self.bias_fixed_mask.to(device=self.bias.device, dtype=self.bias.dtype)
                effective_bias = self.bias * bias_mask
            elif self.bern:
                effective_bias = TCBMaskedParam.apply(self.bias, bias_threshold)
            else:
                bias_mask = (torch.abs(self.bias) >= bias_threshold)
                effective_bias = self.bias * bias_mask

        return F.linear(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True, param_name='weight', fixed_mask=None):
        if param_name == 'weight':
            self.weight_threshold = value
            self.weight_fixed_mask = fixed_mask.detach().clone() if fixed_mask is not None else None
        elif param_name == 'bias':
            self.bias_threshold = value
            self.bias_fixed_mask = fixed_mask.detach().clone() if fixed_mask is not None else None
        else:
            raise ValueError(f'Unsupported parameter name: {param_name}')
        self.bern = bern

    def clear_threshold(self):
        self.weight_threshold = torch.tensor(0.)
        self.bias_threshold = torch.tensor(0.)
        self.weight_fixed_mask = None
        self.bias_fixed_mask = None
        self.bern = True

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
            + (np.prod(self.bias.shape) if self.bias is not None else 0)

class MaskedBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, scaler_rate=1, track=False):
        super(MaskedBlock, self).__init__()
        self.thresholds = OrderedDict()
        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.conv2 = MaskedConv2d(out_channels, int(self.expansion * out_channels),
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(int(self.expansion * out_channels), momentum=None, track_running_stats=track)
        self.shortcut = MaskedConv2d(in_channels, int(self.expansion * out_channels),
                                  kernel_size=1, stride=stride, bias=False)
        self.scaler = Scaler(scaler_rate)

    def forward(self, x):
        shortout = self.shortcut(x[1])  # x[0]: post-activation values; x[1]: pre-activation values.
        out = self.conv1(x[0])
        out = F.relu(self.n1(self.scaler(out)))
        out = self.conv2(out)
        temp = out + shortout
        out1 = out + shortout
        out1 = (F.relu(self.n2(self.scaler(out1)))) * 1.0
        return out1, temp

    def clear_idx(self):
        self.thresholds.clear()


class MaskedResNet(nn.Module):
    def __init__(self, datashape, hidden_size, block, num_blocks, num_classes, scaler_rate=1, track=False, model_name=None):
        super(MaskedResNet, self).__init__()
        self.datashape = datashape
        self.thresholds = OrderedDict()
        self.in_channels = hidden_size[0]
        self.conv = MaskedConv2d(datashape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        self.scaler = Scaler(scaler_rate)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, scaler_rate=scaler_rate,
                                       track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.linear = MaskedLinear(int(hidden_size[3] * block.expansion), num_classes)

    def forward(self, x):
        temp = self.conv(x)
        out = (F.relu(self.n1(self.scaler(self.conv(x)))), temp)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out, _ = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride, scaler_rate, track=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, scaler_rate=scaler_rate, stride=stride, track=track))
            self.in_channels = int(out_channels * block.expansion)
        return nn.Sequential(*layers)

    def get_mask_fiarse(self, rate: float = 0.5, use_tcbgd=True):
        if use_tcbgd is None:
            use_tcbgd = cfg.get('fiarse_use_tcbgd', True)
        self.thresholds.clear()
        for module_name, module in self.named_modules():
            if not isinstance(module, (MaskedConv2d, MaskedLinear)):
                continue
            params = list(module.named_parameters(recurse=False))
            # Select one threshold from the layer's combined weight and bias.
            scores = torch.cat([param.detach().abs().reshape(-1) for _, param in params])
            k = max(1, min(scores.numel(), int(scores.numel() * rate)))
            threshold = torch.topk(scores, k=k).values[-1]
            for tensor_name, param in params:
                fixed_mask = None if use_tcbgd else param.detach().abs() >= threshold
                module.set_threshold(threshold, bern=use_tcbgd,
                                     param_name=tensor_name, fixed_mask=fixed_mask)
                self.thresholds[f'{module_name}.{tensor_name}'] = threshold

    def clear_idx(self):
        self.thresholds.clear()
        for module in self.modules():
            if hasattr(module, 'clear_threshold'):
                module.clear_threshold()
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            for blc in eval(fun_name + s):
                blc.clear_idx()


def masked_resnet18(datashape, hidden_size, num_blocks, num_classes, track=False, model_rate=1, model_name=None):
    hidden_size = [int(item * model_rate) for item in hidden_size]
    model = MaskedResNet(datashape, hidden_size, MaskedBlock, num_blocks, num_classes, model_rate, track, model_name)
    model.apply(init_param)
    return model


if __name__ == '__main__':
    model_half = masked_resnet18([64], [64, 128, 256, 512], [3, 4, 6, 3], 200, model_rate=1)
    for k, v in model_half.state_dict().items():
        print(f"{k}, param shape: {v.shape}")
    print('yes')
