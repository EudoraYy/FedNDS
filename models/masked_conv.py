import torch
import torch.nn as nn
import copy
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
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
        ctx.save_for_backward(param, threshold, mask)   # Save the values needed for the backward pass.
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


class masked_Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        self.thresholds = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.classes_size = classes_size
        self.conv = MaskedConv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = hidden_size[0]
        self.layer1 = self._make_layer(hidden_size[1], track)
        self.layer2 = self._make_layer(hidden_size[2], track)
        self.layer3 = self._make_layer(hidden_size[3], track)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = MaskedLinear(hidden_size[-1], classes_size)

    def _make_layer(self, out_channels, track):
        layer = nn.Sequential(OrderedDict([
            ('maxPool', nn.MaxPool2d(2)),
            ('conv', MaskedConv2d(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('scale', Scaler(self.rate)),
            ('norm', nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.in_channels = out_channels
        return layer

    def forward(self, input):
        out = self.relu(self.norm(self.scaler(self.conv(input))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(self.avgpool(out))
        out = self.linear(out)
        return out


    def get_mask_fiarse(self, rate: float = 0.5, use_tcbgd=None):
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


def masked_conv(data_shape, hidden_size, classes_size, model_rate=1, track=False):
    hidden_size = [int(model_rate * x) for x in hidden_size]
    model = masked_Conv(data_shape, hidden_size, classes_size, model_rate, track)
    model.apply(init_param)
    return model


if __name__ == '__main__':
    data_shape = [3, 24, 24]
    hidden_size = [64, 128, 256, 512]
    model_rate = 1
    classes_size = 10

    x = torch.rand(5, 3, 24, 24)
    model = masked_conv(data_shape, hidden_size, classes_size, model_rate)
    model_half = masked_conv(data_shape, hidden_size, classes_size, 0.5)
    model.get_idx_hetero(0.5)
    param_idx = copy.deepcopy(model.idx)
    params = get_model_params(model, param_idx)
    model_half.load_state_dict(params)

    # for k, v in model.named_modules():
    #     print(k)
    # print(model)
    for k, v in model_half.state_dict().items():
        print(f"{k}, param shape: {v.shape}")

    y = model(x)
    y1 = model_half(x)
    # print(y1.shape)
