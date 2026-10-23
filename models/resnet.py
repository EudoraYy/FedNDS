import math

import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy

if __name__ == '__main__':
    from utils import init_param, Scaler
else:
    from .utils import init_param, Scaler


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, scaler_rate=1, track=False):
        super(Block, self).__init__()
        self.idx = OrderedDict()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # track_running_stats controls the use of running statistics for consistent inference.
        self.n1 = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=int(self.expansion * out_channels), kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(int(self.expansion * out_channels), momentum=None, track_running_stats=track)
        self.shortcut = nn.Conv2d(in_channels, int(self.expansion * out_channels), kernel_size=1, stride=stride,
                                  bias=False)
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

    def get_idx_aware(self, input, first_channels, rate, param_name, topmode):
        shortout = self.shortcut(input[1])
        out = F.relu(self.n1(self.scaler(self.conv1(input[1]))))
        second_channels = get_topk_index(out, int(rate * self.conv1.weight.shape[0]), topmode)
        self.idx[param_name + '.conv1.weight'], self.idx[param_name + '.conv1.bias'] = (second_channels,
                                                                                        first_channels), second_channels
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        out = F.relu(self.n2(self.scaler(self.conv2(out))))
        third_channels = get_topk_index(out, int(rate * self.conv2.weight.shape[0]), topmode)
        self.idx[param_name + '.conv2.weight'], self.idx[param_name + '.conv2.bias'] = (third_channels,
                                                                                        second_channels), third_channels
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        temp = out + shortout
        out += shortout
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
        return (out, temp), third_channels

    def get_idx_roll(self, first_channels, rate, roll, param_name):
        second_channels = torch.roll(torch.arange(self.conv1.weight.shape[0]), shifts=roll % self.conv1.weight.shape[0],
                                     dims=-1)[: int(rate * self.conv1.weight.shape[0])]
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        third_channels = torch.roll(torch.arange(self.conv2.weight.shape[0]), shifts=roll % self.conv2.weight.shape[0],
                                    dims=-1)[: int(rate * self.conv2.weight.shape[0])]
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def get_idx_rand(self, first_channels, rate, param_name, num_selected):
        second_channels = torch.randperm(self.conv1.weight.shape[0])[: int(rate * self.conv1.weight.shape[0])]
        num_selected[param_name + '.conv1.weight'][second_channels] += 1
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels

        third_channels = torch.randperm(self.conv2.weight.shape[0])[: int(rate * self.conv2.weight.shape[0])]
        num_selected[param_name + '.conv2.weight'][third_channels] += 1
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def get_idx_ecover(self, first_channels, rate, param_name, buffer):
        total_num_channels = self.conv1.weight.shape[0]
        second_channels = ecover_choose_idx(buffer, param_name + '.conv1.weight', rate, total_num_channels)
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels

        total_num_channels = self.conv2.weight.shape[0]
        third_channels = ecover_choose_idx(buffer, param_name + '.conv2.weight', rate, total_num_channels)
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def get_idx_hetero(self, first_channels, rate, param_name):
        second_channels = torch.arange(int(rate * self.conv1.weight.shape[0]))
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels

        third_channels = torch.arange(int(rate * self.conv2.weight.shape[0]))
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def clear_idx(self):
        self.idx.clear()


class ResNet(nn.Module):
    def __init__(self, datashape, hidden_size, block, num_blocks, num_classes, scaler_rate=1, track=False, model_name=None):
        super(ResNet, self).__init__()
        self.roll = 0
        self.datashape = datashape
        self.idx = OrderedDict()
        self.in_channels = hidden_size[0]
        self.num_selected = initialize_num_selected_neurons(model_name)

        # self.neuron_selection_count_first_conv_layer = {neuron_idx: 0 for neuron_idx in
        #                                                 np.arange(hidden_size[0])}  # adding

        self.conv = nn.Conv2d(datashape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
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
        self.linear = nn.Linear(int(hidden_size[3] * block.expansion), num_classes)

    def forward(self, x):
        '''
        out = F.relu(self.n1(self.scaler(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        '''
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

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(self.datashape[0]))
        temp = self.conv(input)
        out = (F.relu(self.n1(self.scaler(self.conv(input)))), temp)
        first_channels = get_topk_index(out[0], int(rate * self.conv.weight.shape[0]), topmode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                out, first_channels = blc.get_idx_aware(out, first_channels, rate, param_sub_name, topmode)
                self.idx.update(blc.idx)
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_aware_grad(self, rate, select_mode, gradient):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = get_topk_index(gradient['conv'], int(rate * self.conv.weight.shape[0]), select_mode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            layer_loc = self.get_attar(fun_name+s)
            # for lay in ['0', '1']:
            for lay in range(len(layer_loc)):
                param_name = fun_name + s + '.' + str(lay)
                blc = self.get_attar(param_name)
                second_channels = get_topk_index(gradient[param_name + '.conv1'], int(rate * blc.conv1.weight.shape[0]),
                                                 select_mode)
                self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
                self.idx[param_name + '.n1.weight'], self.idx[
                    param_name + '.n1.bias'] = second_channels, second_channels

                third_channels = get_topk_index(gradient[param_name + '.conv2'], int(rate * blc.conv2.weight.shape[0]),
                                                select_mode)
                self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
                self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
                self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
                first_channels = third_channels
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_rand_grad(self, rate, select_mode, gradient, epoch):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = get_topk_index(gradient['conv.weight'], int(rate * self.conv.weight.shape[0]), select_mode,
                                        epoch, self.num_selected['conv.weight'])
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_first_conv_layer[neuron_idx.item()] += 1
        self.num_selected['conv.weight'][first_channels] += 1
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            layer_loc = self.get_attar(fun_name+s)
            # for lay in ['0', '1']:
            for lay in range(len(layer_loc)):
                param_name = fun_name + s + '.' + str(lay)
                blc = self.get_attar(param_name)
                second_channels = get_topk_index(gradient[param_name + '.conv1.weight'], int(rate * blc.conv1.weight.shape[0]),
                                                 select_mode, epoch, self.num_selected[param_name + '.conv1.weight'])
                self.num_selected[param_name + '.conv1.weight'][second_channels] += 1
                self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
                self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
                third_channels = get_topk_index(gradient[param_name + '.conv2.weight'], int(rate * blc.conv2.weight.shape[0]),
                                                select_mode, epoch, self.num_selected[param_name + '.conv2.weight'])
                self.num_selected[param_name + '.conv2.weight'][third_channels] += 1
                self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
                self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
                self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
                first_channels = third_channels
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_roll(self, rate, roll_offset=None):
        roll = self.roll if roll_offset is None else roll_offset
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.roll(torch.arange(self.conv.weight.shape[0]),
                                    shifts=roll % self.conv.weight.shape[0], dims=-1)[
                         : int(rate * self.conv.weight.shape[0])]
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_roll(first_channels, rate, roll, param_sub_name)
                self.idx.update(blc.idx)
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.randperm(self.conv.weight.shape[0])[: int(rate * self.conv.weight.shape[0])]
        self.num_selected['conv.weight'][first_channels] += 1
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_rand(first_channels, rate, param_sub_name, self.num_selected)
                self.idx.update(blc.idx)
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_ecover(self, rate, model_name=None):
        if not hasattr(self, '_ecover_buffer'):
            self._ecover_buffer = initialize_unused_param_indices_for_layers(model_name)
        start_channels = (torch.arange(self.datashape[0]))
        total_num_channels = self.conv.weight.shape[0]
        layer_name = 'conv.weight'
        first_channels = ecover_choose_idx(self._ecover_buffer, layer_name, rate, total_num_channels)
        self.idx[layer_name] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_ecover(first_channels, rate, param_sub_name, self._ecover_buffer)
                self.idx.update(blc.idx)
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.arange(int(rate * self.conv.weight.shape[0]))
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_hetero(first_channels, rate, param_sub_name)
                self.idx.update(blc.idx)
        # for neuron_idx in first_channels:
        #     self.neuron_selection_count_last_conv_layer[neuron_idx.item()] += 1
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_aware_layer_wise(self, input, rate, topmode, device):
        start_channels = (torch.arange(self.datashape[0]))
        temp = self.conv(input)
        out = F.relu(self.n1(self.scaler(self.conv(input))))
        first_channels = self.topk(out, 'conv', int(rate * self.conv.weight.shape[0]), topmode, device)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'layer'
        seq_index = ['1', '2', '3', '4']
        # second_channels, third_channels = torch.tensor([]), torch.tensor([])
        for s in seq_index:
            for lay in ['0', '1']:
                param_name = fun_name + s + '.' + lay
                blc = self.get_attar(param_name)
                shortout = blc.shortcut(temp)
                out = F.relu(blc.n1(blc.scaler(blc.conv1(out))))
                second_channels = self.topk(out, param_name + '.conv1', int(rate * blc.conv1.weight.shape[0]), topmode,
                                            device)
                self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
                self.idx[param_name + '.n1.weight'], self.idx[
                    param_name + '.n1.bias'] = second_channels, second_channels
                out = F.relu(blc.n2(blc.scaler(blc.conv2(out))))
                third_channels = self.topk(out, param_name + '.conv2', int(rate * blc.conv2.weight.shape[0]), topmode,
                                           device)
                self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
                self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
                temp = out + shortout
                out += shortout
                self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
                first_channels = third_channels
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def topk(self, input, name, k, topmode, device):
        names = name.split('.')
        # pre_sum = torch.sum(input)
        if names[0] == 'conv1':
            blc = self.get_attar('layer1.0')
            return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
        elif names[0][:-1] == 'layer':
            attr_name = names[0]
            for s in names[1:-1]:
                attr_name += ('.' + s)
            if names[-1] == 'conv1':
                blc = self.get_attar(attr_name)
                return self.get_diff_channels_sum(input, blc, 2, k, topmode, device)
            elif names[-1] == 'conv2':
                if names[1] == '0':
                    blc = self.get_attar(attr_name.replace('0', '1'))
                    return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
                elif names[1] == '1':
                    if names[0][-1] == '4':
                        blc = self.get_attar('linear')
                        return self.get_diff_channels_sum(input, blc, 3, k, topmode, device)
                    else:
                        next_layer = str(int(names[0][-1]) + 1)
                        names[0] = names[0][:-1] + next_layer
                        names[1] = '0'
                        attr_name_temp = names[0]
                        for s in names[1:-1]:
                            attr_name_temp += ('.' + s)
                        blc = self.get_attar(attr_name_temp)
                        return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
                else:
                    raise ValueError('layer.0 or 1 error')
        else:
            raise ValueError('names[0]'.format(names[0]))

    def get_diff_channels_sum(self, input, blc, conv_index, k, topmode, device):
        # x = copy.deepcopy(input)
        res = []
        if conv_index == 1 or conv_index == 2:
            if conv_index == 1:
                conv = blc.conv1
                n = blc.n1
            elif conv_index == 2:
                conv = blc.conv2
                n = blc.n2
            else:
                raise ValueError('not match conv')
            for i in range(input.shape[1] + 1):
                # temp = copy.deepcopy(input)
                temp = input.clone()
                if i != input.shape[1]:
                    temp[:, i, :, :] = 0
                i_sum = F.relu(n(blc.scaler(conv(temp))))
                i_sum = torch.sum(i_sum)
                res.append(i_sum)
            pre_sum = res.pop()
            pre_sum = pre_sum.to(device)
            diff_channels_sum = torch.tensor(res).to(device)
            diff_channels_sum = torch.abs(diff_channels_sum - pre_sum)
            return get_topk_index(diff_channels_sum, k, topmode)
        else:  # layer4.1.conv2  using Linear
            for i in range(input.shape[1] + 1):
                # temp = copy.deepcopy(input)
                temp = input.clone()
                if i != input.shape[1]:
                    temp[:, i, :, :] = 0
                temp = F.adaptive_avg_pool2d(temp, 1)
                temp = temp.view(temp.size(0), -1)
                temp = blc(temp)
                i_sum = torch.sum(temp)
                res.append(i_sum)
            pre_sum = res.pop()
            pre_sum = pre_sum.to(device)
            diff_channels_sum = torch.tensor(res).to(device)
            diff_channels_sum = torch.abs(diff_channels_sum - pre_sum)
            return get_topk_index(diff_channels_sum, k, topmode)

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj

    def clear_idx(self):
        self.idx.clear()
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            for blc in eval(fun_name + s):
                blc.clear_idx()


def initialize_num_selected_neurons(model_name):
    num_selected_neurons_for_layers = {}
    layers = ["layer1", "layer2", "layer3", "layer4"]
    out_channel_numbers = [64, 128, 256, 512]
    if model_name == "resnet18":
        blocks = ["0", "1"]
        num_selected_neurons_for_layers["conv.weight"] = torch.zeros_like(torch.arange(64))
        for i, layer in enumerate(layers):
            for block in blocks:
                num_selected_neurons_for_layers[f"{layer}.{block}.conv1.weight"] = torch.zeros_like(torch.arange(out_channel_numbers[i]))
                num_selected_neurons_for_layers[f"{layer}.{block}.conv2.weight"] = torch.zeros_like(torch.arange(out_channel_numbers[i]))
    elif model_name == "resnet34":
        blocks = [["0", "1", "2"], ["0", "1", "2", "3"], ["0", "1", "2", "3", "4", "5"], ["0", "1", "2"]]
        num_selected_neurons_for_layers["conv.weight"] = torch.zeros_like(torch.arange(64))
        for i, layer in enumerate(layers):
            for block in blocks[i]:
                num_selected_neurons_for_layers[f"{layer}.{block}.conv1.weight"] = torch.zeros_like(torch.arange(out_channel_numbers[i]))
                num_selected_neurons_for_layers[f"{layer}.{block}.conv2.weight"] = torch.zeros_like(torch.arange(out_channel_numbers[i]))
    return num_selected_neurons_for_layers


def initialize_unused_param_indices_for_layers(model_name):
    unused_param_indices_for_layers = {}
    layers = ["layer1", "layer2", "layer3", "layer4"]
    out_channel_numbers = [64, 128, 256, 512]
    if model_name == "resnet18":
        blocks = ["0", "1"]
        unused_param_indices_for_layers["conv.weight"] = torch.from_numpy(np.arange(64))
        for i, layer in enumerate(layers):
            for block in blocks:
                unused_param_indices_for_layers[f"{layer}.{block}.conv1.weight"] = torch.from_numpy(np.arange(out_channel_numbers[i]))
                unused_param_indices_for_layers[f"{layer}.{block}.conv2.weight"] = torch.from_numpy(np.arange(out_channel_numbers[i]))
    elif model_name == "resnet34":
        blocks = [["0", "1", "2"], ["0", "1", "2", "3"], ["0", "1", "2", "3", "4", "5"], ["0", "1", "2"]]
        unused_param_indices_for_layers["conv.weight"] = torch.from_numpy(np.arange(64))
        for i, layer in enumerate(layers):
            for block in blocks[i]:
                unused_param_indices_for_layers[f"{layer}.{block}.conv1.weight"] = torch.from_numpy(np.arange(out_channel_numbers[i]))
                unused_param_indices_for_layers[f"{layer}.{block}.conv2.weight"] = torch.from_numpy(np.arange(out_channel_numbers[i]))
    return unused_param_indices_for_layers


def ecover_choose_idx(buffer, layer, rate, total_num_channels):
    sample_num = int(rate * total_num_channels)
    buffer_size = buffer[layer].size(0)
    if sample_num > buffer_size:
        first_channels = buffer[layer].numpy().copy()
        remaining_sample_num = sample_num - first_channels.size
        pool = np.arange(total_num_channels)
        additional_channels = np.random.choice(np.setdiff1d(pool, first_channels),
                                               remaining_sample_num, replace=False)
        first_channels = np.sort(np.concatenate((first_channels, additional_channels)))
        buffer[layer] = torch.from_numpy(np.setdiff1d(pool, additional_channels))
    else:
        first_channels = np.sort(np.random.choice(buffer[layer], sample_num, replace=False))
        buffer[layer] = torch.from_numpy(np.setdiff1d(buffer[layer], first_channels))
    return torch.tensor(first_channels)


def get_idx_sorted_gradient(x, topmode):
    if topmode == 'l2':
        if x.dim() > 2:  # conv
            temp = torch.norm(x, p=2, dim=(1, 2, 3))
            gradient_sorted_idx = torch.argsort(temp, descending=True)
            return gradient_sorted_idx
        else:  # linear
            gradient_sorted_idx = torch.argsort(x, descending=True)
            return gradient_sorted_idx


def get_topk_index(x, k, topmode, epoch=1, num_selected=None):
    if topmode == 'absmax':
        if x.dim() > 2:  # conv
            temp = torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:  # linear
            return torch.topk(x, k)[1]
    elif topmode == 'l2':
        if x.dim() > 2:  # conv
            temp = torch.square(torch.norm(x, p=2, dim=(1, 2, 3)))
            # additive_term = torch.zeros_like(num_selected, dtype=torch.float32)
            # for i in range(len(num_selected)):
            #     additive_term[i] = (torch.mean(temp) / 100) \
            #                        * math.sqrt(torch.sum(num_selected) * math.log(epoch) / num_selected[i])
            # indicator = temp + math.exp(0.001 * epoch) * additive_term
            # print("additive term: {}".format(additive_term))
            # print("updates_norm: {}".format(temp))
            # print("indicator: {}".format(indicator))
            return torch.topk(temp, k)[1]
        else:  # linear
            # additive_term = torch.zeros_like(num_selected, dtype=torch.float32)
            # for i in range(len(num_selected)):
            #     additive_term[i] = (torch.mean(x) / 100) \
            #                        * math.sqrt(torch.sum(num_selected) * math.log(epoch) / num_selected[i])
            # indicator = x + math.exp(0.001 * epoch) * additive_term
            return torch.topk(x, k)[1]
    elif topmode == 'probs':
        if x.dim() > 2:
            temp = torch.sum(x, dim=(0, 2, 3))
            probs = torch.abs(temp) / torch.sum(torch.abs(temp))
        else:
            probs = torch.abs(x) / torch.sum(torch.abs(x))
        samples = torch.multinomial(probs, num_samples=k, replacement=False)
        return samples
    elif topmode == 'absmin':
        if x.dim() > 2:
            temp = -torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:
            return torch.topk(-x, k)[1]
    else:
        raise ValueError('no method!')


def get_model_params(global_model, client_model_idx, mode=None):
    client_model_params = OrderedDict()
    if mode == 'fiarse':
        for k, v in global_model.state_dict().items():
            client_model_params[k] = v * client_model_idx[k]
    else:
        for k, v in global_model.state_dict().items():
            if k in client_model_idx:
                if v.dim() > 1:
                    client_model_params[k] = copy.deepcopy(v[torch.meshgrid(client_model_idx[k], indexing='ij')])
                else:
                    client_model_params[k] = copy.deepcopy(v[client_model_idx[k]])
            else:
                raise NameError('Can\'t match {}'.format(k))
    return client_model_params


def resnet18(datashape, hidden_size, num_blocks, num_classes, track=False, model_rate=1, model_name=None):
    hidden_size = [int(item * model_rate) for item in hidden_size]
    model = ResNet(datashape, hidden_size, Block, num_blocks, num_classes, model_rate, track, model_name)
    model.apply(init_param)
    return model


if __name__ == '__main__':
    # rate = 0.5
    # x = torch.rand((4, 3, 25, 25))
    # model = resnet18([3], [64, 128, 256, 512], [3, 4, 6, 3], 10, model_rate=1)
    # print(model(x).shape)
    model_half = resnet18([64], [64, 128, 256, 512], [3, 4, 6, 3], 200, model_rate=1)
    # model.get_idx_aware(x, 0.5, 'absmax')
    # model_half_shape = copy.deepcopy(model.idx)
    # model.clear_idx()
    # model_half_params = OrderedDict()
    for k, v in model_half.state_dict().items():
        print(f"{k}, param shape: {v.shape}")
    print('yes')
